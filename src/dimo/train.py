"""Server entry point: strict origin -> GPU preflight -> uninterrupted DiMO."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, InitProcessGroupKwargs, DistributedType
from src.dimo.device import initialize_local_cuda_device
from src.dimo.origin import select_teacher, load_strict, sha256, weight_files
from src.dimo.core import (clone_models, generation_mask, predict, distillation_losses,
                           auxiliary_ce, update_ema, generate_one_step, pseudo_mask)
from src.dimo.data import ReplayStream
from src.dimo.checkpoint import (collective_call, save_checkpoint, load_checkpoint,
                                  check_roundtrip, read_progress, verify_checkpoint, state_digest,
                                  validate_progress, microbatches_per_update, prune_checkpoints)
from src.timestep_sampling import MaskRatioSampler
from src.mask_utils import MAGICBRUSH_MASK_DECODING
from src.dimo.telemetry import Telemetry, grad_norm
from src.dimo.config import resolve_config, validate_config, source_metadata
from src.dimo.optimization import configure_trainability, build_optimizers, build_schedulers


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def latest_checkpoint(out):
    candidates = [(int(m[1]), p) for p in out.glob('checkpoint-*')
                  if p.is_dir() and (m := re.fullmatch(r'checkpoint-(\d+)', p.name))
                  and (p / 'COMPLETE').is_file()]
    if not candidates:
        raise ValueError(f'No complete DiMO checkpoint in {out}')
    selected = max(candidates)[1]
    verify_checkpoint(selected)  # Corruption is a blocking error; never silently roll back.
    return selected


def dataset_stream(c, tokenizer, acc, consumed):
    if c.get('data_manifest'):
        from src.dimo.manifest_data import manifest_stream
        return manifest_stream(c, tokenizer, acc, consumed)
    from src.inter_edit_dataset import InterEditDataset
    from src.magicbrush_dataset import MagicBrushDataset
    from src.mixed_edit_dataset import MixedEditDataset
    common = dict(tokenizer=tokenizer, size=c['resolution'], text_encoder_architecture='CLIP_Gemma2',
                  rank=acc.process_index, world_size=acc.num_processes, seed=c['seed'], language=c['language'])
    inter = InterEditDataset(root_dir=c['inter_edit_root'], split='train', cache_dir=c['cache_dir'], **common)
    magic = MagicBrushDataset(root_dir=c['magicbrush_root'], split='train',
                             cache_dir=Path(c['cache_dir']) / 'magicbrush', **common)
    ds = MixedEditDataset([inter, magic], ['Inter-Edit', 'MagicBrush'],
                          [1 - c['magicbrush_probability'], c['magicbrush_probability']],
                          rank=acc.process_index, seed=c['seed'])
    # Readers already shard by rank. Do not Accelerator.prepare this loader.
    loader = torch.utils.data.DataLoader(ds, batch_size=c['micro_batch'], num_workers=0)
    return ReplayStream(loader, c['seed'] + acc.process_index, consumed)


def prepare_next_batch(acc, stream, encoders, vq, empty, teacher):
    """Read and prepare one microbatch with a single all-rank error exchange."""
    def action():
        batch = next(stream)
        return batch, prepare_batch(batch, encoders, vq, empty, teacher, acc)
    return collective_call(acc, action)


def prepare_batch(batch, encoders, vq, empty, teacher, acc):
    from src.v2_utils import prepare_cond_token
    from src.dataset_utils import move_batch_prompt_device, encode_prompt
    from src.mask_utils import resize_edit_mask_to_tokens
    from src.pipeline import _prepare_latent_image_ids
    with torch.no_grad():
        source = prepare_cond_token(None, batch['reference_image'].to(acc.device), vq)
        b, n = source.shape
        side = int(n ** .5)
        source = source.reshape(b, side, side)
        original = resize_edit_mask_to_tokens(batch['edit_mask'].to(acc.device), (side, side)).reshape_as(source)
        region = generation_mask(original, teacher)
        prompts = move_batch_prompt_device(batch['prompt_input_ids'], acc.device, split=True)
        hidden, pooled = encode_prompt(encoders, prompts, 'CLIP_Gemma2')
        ids = _prepare_latent_image_ids(side, side, acc.device, source.dtype,
                                       up_sample=teacher.down_block.downsample is None)
        cond = dict(encoder_hidden_states=hidden, pooled_projections=pooled,
                    micro_conds=batch['micro_conds'].to(acc.device), img_ids=ids,
                    txt_ids=torch.zeros(hidden.shape[1], 3, device=acc.device, dtype=source.dtype),
                    target_edit_mask=original, reference_image_hidden_states=source,
                    reference_image_ids=ids, lora_part_enable=getattr(teacher, '_dimo_lora_part_enable', True))
        uncond = dict(cond, encoder_hidden_states=empty[0].expand_as(hidden),
                      pooled_projections=empty[1].expand_as(pooled))
    # batch['image'] (GT) never participates in distillation or VQ encoding.
    return source, original, region, cond, uncond


def require_finite(value,message):
    # On CUDA report device assertions at the next scheduled synchronization;
    # avoid a host scalar transfer on every microbatch. CPU errors are immediate.
    finite=torch.isfinite(value).all()
    if value.is_cuda:torch._assert_async(finite,message)
    elif not finite:raise RuntimeError(message)


def assert_gradients(model, label, required):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not required and grads:
        raise RuntimeError(f'{label}: gradient isolation failed')
    if required and (not grads or not all(torch.isfinite(g).all() for g in grads)
                     or not any(g.count_nonzero() for g in grads)):
        raise RuntimeError(f'{label}: missing, zero or nonfinite gradients')


def run_update(acc, teacher, models, optimizers, schedulers, batch_fn, sampler, c, preflight,
               global_step=None, telemetry=None, extensions=None):
    """One Student update, then fresh-data Auxiliary updates, then Student EMA.

    Mirrors official train_DIMO_Meissonic.py generator/fake_model phases.
    No Student-phase tokens, corruption masks or batches are reused for Aux CE.
    """
    if c.get('ema_warmup', False) and (type(global_step) is not int or global_step < 0):
        raise ValueError('EMA warmup requires the restored zero-based Student global_step')
    student, auxiliary, ema = (models[k] for k in ('student', 'auxiliary', 'ema'))
    accumulation = c['gradient_accumulation']
    fake_rounds = c.get('fake_rounds', 1)
    consumed = microbatches_per_update(c)
    r_init = c.get('student_initial_mask_ratio', 1.0)
    label_smoothing = c.get('auxiliary_label_smoothing', 0.1)
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
    losses = dict(student=[], auxiliary=[], microbatches=consumed)
    if extensions is not None:
        if global_step is not None and extensions.phase_step!=global_step:raise ValueError('Optional loss phase counter must follow this new training stage')
        extensions.records={}

    cuda = acc.device.type == 'cuda'
    if cuda:torch.cuda.reset_peak_memory_stats(acc.device)
    # Phase 1: accumulate FKL gradients, step only Student, then clear its grads.
    for _ in range(accumulation):
        source, original, region, cond, uncond = batch_fn()
        ratio = sampler.sample(len(source), acc.device)
        with acc.autocast():
            options = dict(diagnostics=telemetry.observe) if telemetry is not None and (telemetry.diagnostic_due or telemetry.token_due) else {}
            active=extensions is not None and extensions.student_active()
            from src.dimo.losses import capture_logits
            with capture_logits(student) if active else nullcontext(None) as captured:
                loss_s, payload = distillation_losses(teacher, student, auxiliary, source, original,
                    region, teacher.config.vocab_size - 1, cond, uncond, ratio, r_init=r_init,
                    teacher_cfg=c.get('teacher_cfg',4.0), noise_emb_perturb=c.get('noise_emb_perturb',0.0), **options)
            if active:
                if len(captured)!=1:raise RuntimeError('Optional losses must reuse one Student forward')
                raw_logits=captured.pop()
                loss_s=extensions.student_loss(loss_s,raw_logits.movedim(1,-1),
                    (source,original,region,cond,uncond),raw_logits=raw_logits)
            elif extensions is not None:extensions.records.setdefault('fkl',[]).append(loss_s.detach())
        require_finite(loss_s,'Nonfinite Student loss')
        acc.backward(loss_s / accumulation)
        if preflight:
            assert_gradients(teacher, 'Teacher', False)
            assert_gradients(ema, 'EMA', False)
            assert_gradients(auxiliary, 'Aux during Student backward', False)
            if extensions is not None and hasattr(extensions,'discriminator'):
                assert_gradients(extensions.discriminator,'Discriminator during Student backward',False)
            assert_gradients(student, 'Student', True)
            _, sampled, q, _ = payload
            if not torch.equal(sampled[~region], source[~region]) or (q & ~region).any():
                raise RuntimeError('Source token hard lock / pseudo-mask subset failed')
        losses['student'].append(loss_s.detach())
        if active:del raw_logits
    # Official generator clipping is conditional on mixed precision. Aux has no clipping.
    norm = None
    if '16' in c.get('mixed_precision', 'no') and c.get('max_grad_norm', 1.) > 0:
        norm = acc.clip_grad_norm_(student.parameters(), c.get('max_grad_norm', 1.))
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite Student gradient norm')
    if telemetry is not None and (telemetry.loss_due or telemetry.grad_due):
        if norm is None:norm=grad_norm(student)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite Student gradient norm')
        if telemetry.grad_due:telemetry.add_sum('grad/student_pre_clip_l2',norm)
    optimizers['student'].step()
    optimizers['student'].zero_grad(set_to_none=True)
    schedulers['student'].step()

    losses['student_peak_bytes']=torch.cuda.max_memory_allocated(acc.device) if cuda else None
    if cuda:torch.cuda.reset_peak_memory_stats(acc.device)
    # Phase 2: keep consuming the SAME stream at NEW positions. Generate targets
    # from the UPDATED Student under no_grad, then sample a NEW pseudo-mask.
    for _ in range(fake_rounds):
        optimizers['auxiliary'].zero_grad(set_to_none=True)
        round_losses = []
        for _ in range(accumulation):
            source, original, region, cond, _ = batch_fn()
            with acc.autocast():
                with torch.no_grad():
                    if extensions is not None and hasattr(extensions,'discriminator'):
                        logits,sampled=generate_one_step(acc.unwrap_model(student),source,original,
                            teacher.config.vocab_size-1,cond,r_init=r_init,return_logits=True,
                            noise_emb_perturb=c.get('noise_emb_perturb',0.0))
                        extensions.remember_auxiliary(logits,sampled,(source,original,region,cond,None))
                        del logits
                    else:
                        sampled = generate_one_step(acc.unwrap_model(student), source, original,
                            teacher.config.vocab_size - 1, cond, r_init=r_init,
                            noise_emb_perturb=c.get('noise_emb_perturb',0.0))
                    if telemetry is not None and telemetry.token_due:
                        telemetry.tokens('aux_targets',sampled,region)
                    ratio = sampler.sample(len(source), acc.device)
                    q = pseudo_mask(region, ratio)
                    corrupted = torch.where(q, teacher.config.vocab_size - 1, sampled)
                loss_a = auxiliary_ce(predict(auxiliary, corrupted, ratio, cond), sampled, q,
                                      label_smoothing=label_smoothing)
            require_finite(loss_a,'Nonfinite Auxiliary loss')
            acc.backward(loss_a / accumulation)
            if preflight:
                assert_gradients(auxiliary, 'Auxiliary', True)
                assert_gradients(student, 'Student during Aux backward', False)
                assert_gradients(teacher, 'Teacher after Aux', False)
                assert_gradients(ema, 'EMA after Aux', False)
                if not torch.equal(sampled[~region], source[~region]) or (q & ~region).any():
                    raise RuntimeError('Source token hard lock / pseudo-mask subset failed')
            round_losses.append(loss_a.detach())
        if telemetry is not None and (telemetry.loss_due or telemetry.grad_due):
            aux_norm=grad_norm(auxiliary)
            if not torch.isfinite(aux_norm):raise RuntimeError('Nonfinite Auxiliary gradient norm')
            if telemetry.grad_due:telemetry.add_sum('grad/auxiliary_unclipped_l2',aux_norm)
        optimizers['auxiliary'].step()
        optimizers['auxiliary'].zero_grad(set_to_none=True)
        schedulers['auxiliary'].step()
        losses['auxiliary'].append(round_losses)
    losses['auxiliary_peak_bytes']=torch.cuda.max_memory_allocated(acc.device) if cuda else None
    if extensions is not None and hasattr(extensions,'discriminator'):
        extensions.discriminator_update(acc,models,optimizers,schedulers,preflight)
    losses['ema_decay'] = update_ema(ema, acc.unwrap_model(student), c['ema_decay'],
                                    global_step=(global_step+c.get('parent_metadata',{}).get('ema_age_base',0)) if c.get('ema_warmup', False) else None)
    if extensions is not None:
        extensions.finish_update()
        losses.update(extensions.records)
    return losses


@torch.no_grad()
def preview(acc, out, step, ema, vq, source, original, cond, batch, r_init=1.0):
    """Save the direct VQ-decoded one-step image for a training sample."""
    from torchvision.utils import save_image
    with acc.autocast():
        tokens = generate_one_step(ema, source, original, ema.config.vocab_size - 1, cond, r_init=r_init)
    b, h, w = tokens.shape
    decoded = vq.decode(tokens.flatten(), force_not_quantize=True,
                        shape=(b, h, w, vq.config.latent_channels)).sample.float().clamp(0, 1)
    save_image(decoded, out / 'validation' / f'step-{step}-rank{acc.process_index}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', choices=['latest'])
    parser.add_argument('--stop-after-preflight', action='store_true')
    parser.add_argument('--max-steps', type=int, help='Extend the constant-LR budget on resume')
    args = parser.parse_args()
    c = resolve_config(args.config)
    if args.max_steps is not None:
        if not args.resume or args.max_steps < c['max_steps']:
            parser.error('--max-steps is only for extending a resumed run')
        c['max_steps'] = args.max_steps
    validate_config(c)
    from src.dimo.initialization import parent_metadata,parent_teacher,load_parent_weights
    parent_metadata(c)
    from src.dimo.infer import validate_base_components
    validate_base_components(c['base_model_dir'])
    c['source_metadata'] = source_metadata()
    initialize_local_cuda_device()
    acc = Accelerator(mixed_precision=c['mixed_precision'], dynamo_backend='no', kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True),
        InitProcessGroupKwargs(timeout=timedelta(minutes=30))])
    if acc.device.type != 'cuda':
        raise RuntimeError('Formal preflight requires CUDA; use pytest for tiny CPU/Gloo verification')
    if acc.num_processes != int(os.environ.get('DIMO_NUM_PROCESSES', acc.num_processes)):
        raise RuntimeError('Launcher process count differs from Accelerator world size')
    if acc.distributed_type not in (DistributedType.NO, DistributedType.MULTI_GPU):
        raise ValueError('Formal checkpoint format requires the configured CUDA/DDP launch mode')
    set_seed(c['seed'] + acc.process_index)
    out = Path(c['output_dir']).resolve()
    # One job owns the output for its lifetime; rank 0 broadcasts lock/I/O errors.
    import fcntl as file_lock
    output_lock = None
    def claim_output():
        nonlocal output_lock
        if not args.resume and out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Output is nonempty: {out}; use --resume latest or a new output directory')
        out.mkdir(parents=True, exist_ok=True)
        output_lock = (out / '.dimo.lock').open('a')
        file_lock.flock(output_lock, file_lock.LOCK_EX | file_lock.LOCK_NB)
    collective_call(acc, claim_output, main_only=True)
    resume = collective_call(acc, lambda: latest_checkpoint(out)) if args.resume else None
    for name in ('preflight', 'validation', 'logs'):
        (out / name).mkdir(exist_ok=True)
    hardware = dict(device=torch.cuda.get_device_name(acc.device), visible_gpus=torch.cuda.device_count(),
                    actual_processes=acc.num_processes, rank=acc.process_index,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    total_memory_bytes=torch.cuda.get_device_properties(acc.device).total_memory,
                    global_batch_size=c['micro_batch'] * c['gradient_accumulation'] * acc.num_processes,
                    effective_batch=c['micro_batch']*c['gradient_accumulation']*acc.num_processes,
                    micro_batch=c['micro_batch'],gradient_accumulation=c['gradient_accumulation'],
                    mixed_precision=c['mixed_precision'])
    write_json(out / 'logs' / f'hardware-rank{acc.process_index}.json', hardware)
    acc.print(hardware)
    torch.cuda.reset_peak_memory_stats()
    if resume:
        state = read_progress(resume, acc.process_index)
        validate_progress(state, c, acc.num_processes)
        origin = read_json(out / 'teacher_origin.json')
        for name, digest in origin['pinned_teacher_checksums'].items():
            if sha256(out / 'teacher' / name) != digest:
                raise ValueError('Pinned output Teacher checksum changed')
        model = load_strict(out / 'teacher')
        step, consumed = state['global_step'], state['consumed_batches']
    else:
        def pin_teacher():
            # Select once, even if the SFT job publishes a newer checkpoint while
            # other ranks start. Every rank loads this same immutable output copy.
            model, origin = parent_teacher(c) if c.get('initialization')=='finetune' else select_teacher(c)
            halo = c.get('training_sparse_target_halo', model.config.sparse_target_halo)
            supervise = c.get('training_supervise_target_halo', model.config.supervise_target_halo)
            model.enable_sparse_target(halo=halo, supervise_halo=supervise)
            origin['runtime_region_profile'] = dict(sparse_target_halo=halo,
                                                    supervise_target_halo=supervise)
            origin['source_metadata']=c['source_metadata']
            origin['source_archive'] = c.get('source_archive')
            origin['source_archive_sha256'] = c.get('source_archive_sha256')
            model.save_pretrained(out / 'teacher', safe_serialization=True)
            origin['pinned_teacher_checksums'] = {p.name: sha256(p) for p in [
                out / 'teacher/config.json', *weight_files(out / 'teacher')]}
            write_json(out / 'teacher_origin.json', origin)
            # JSON is a YAML 1.2 subset.
            write_json(out / 'config_resolved.yaml', c)
        collective_call(acc, pin_teacher, main_only=True)
        model = collective_call(acc, lambda: load_strict(out / 'teacher'))
        step, consumed = 0, 0
    acc.wait_for_everyone()
    teacher, student, auxiliary, ema = clone_models(model)
    if not resume and c.get('initialization')=='finetune':
        collective_call(acc,lambda:load_parent_weights(c,dict(student=student,auxiliary=auxiliary,ema=ema)))
    teacher._dimo_lora_part_enable = c.get('lora_part_enable', True)
    parameter_report=configure_trainability(dict(teacher=teacher,student=student,auxiliary=auxiliary,ema=ema),c)
    collective_call(acc,lambda:write_json(out/'parameter_report.json',parameter_report),main_only=True)
    acc.print(parameter_report)
    del model
    if c['gradient_checkpointing']:
        student.enable_gradient_checkpointing()
        auxiliary.enable_gradient_checkpointing()
    teacher.to(acc.device)
    ema.to(acc.device)
    optimizers=build_optimizers(dict(student=student,auxiliary=auxiliary),c)
    schedulers=build_schedulers(optimizers,c)
    student,auxiliary,opt_s,opt_a=acc.prepare(student,auxiliary,optimizers['student'],optimizers['auxiliary'])
    models=dict(student=student,auxiliary=auxiliary,ema=ema)
    optimizers=dict(student=opt_s,auxiliary=opt_a)
    from src.v2_model import init_text_encoder, init_base_model
    from src.dataset_utils import tokenize_prompt, encode_prompt
    base = Path(c['base_model_dir'])
    for subfolder in ('text_encoder', 'tokenizer', 'llm_encoder', 'vqvae', 'scheduler'):
        if not (base / subfolder).is_dir():
            raise FileNotFoundError(f'Missing local base component: {base / subfolder}')
    encoders, tokenizer = init_text_encoder('CLIP_Gemma2', str(base))
    vq, scheduler = init_base_model(str(base))
    for component in encoders:
        component.requires_grad_(False).eval().to(acc.device, dtype=(
            torch.bfloat16 if c['mixed_precision'] == 'bf16' else torch.float32))
    vq.requires_grad_(False).eval().to(acc.device)
    if vq.config.num_vq_embeddings != teacher.config.codebook_size:
        raise ValueError('Loaded VQ codebook differs from Teacher')
    if scheduler.config.mask_token_id != teacher.config.vocab_size - 1:
        raise ValueError('Loaded Scheduler mask token ID differs from Teacher')
    with torch.no_grad():
        empty_ids = tokenize_prompt(tokenizer, '', 'CLIP_Gemma2', device=acc.device)
        empty = encode_prompt(encoders, empty_ids, 'CLIP_Gemma2')
    stream = dataset_stream(c, tokenizer, acc, consumed)
    from src.dimo.losses import build_losses
    extensions=build_losses(c,vq,teacher,acc.device)
    if extensions is not None and hasattr(extensions,'discriminator'):
        extensions.attach(acc,models,optimizers,schedulers)
    if resume:
        load_checkpoint(acc, resume, c, models, optimizers, schedulers,extensions=extensions)
    sampler = MaskRatioSampler()
    last_batch = {}
    probe_path=out/'validation'/f'fixed-probe-rank{acc.process_index}.pt'
    probe=torch.load(probe_path,map_location=acc.device,weights_only=True) if resume and probe_path.is_file() else None
    def batch_fn():
        nonlocal probe
        batch, prepared = prepare_next_batch(acc, stream, encoders, vq, empty, teacher)
        last_batch['batch'], last_batch['prepared'] = batch, prepared
        if extensions is not None:extensions.bind_batch(batch,prepared)
        if probe is None:
            source,original,_,cond,_=prepared
            probe=dict(source=source.detach(),original=original.detach(),
                       cond={k:v.detach() if torch.is_tensor(v) else v for k,v in cond.items()})
            torch.save(probe,probe_path)
        return prepared
    import time
    while step < c['max_steps']:
        calibration_due=extensions is not None and extensions.needs_calibration()
        diagnostic_due=extensions is not None and hasattr(extensions,'needs_diagnostic') and extensions.needs_diagnostic()
        if calibration_due or diagnostic_due:
            from src.dimo.calibration import calibrate
            shadow=dataset_stream(c,tokenizer,acc,stream.consumed)
            def calibration_batch():
                raw,prepared=prepare_next_batch(acc,shadow,encoders,vq,empty,teacher)
                extensions.bind_batch(raw,prepared)
                return prepared
            result=calibrate(acc,teacher,models,calibration_batch,sampler,c,extensions,update_weight=calibration_due)
            label='calibration' if calibration_due else 'gradient-diagnostic'
            collective_call(acc,lambda:write_json(out/f'{label}-step{step}.json',result),main_only=True)
        started = time.monotonic()
        checking = step < (state['global_step'] + c['preflight_steps'] if resume else c['preflight_steps'])
        telemetry=Telemetry(c,step,acc.device,teacher.config.codebook_size)
        try:
            losses = run_update(acc, teacher, models, optimizers, schedulers, batch_fn, sampler, c, checking,
                                global_step=step,telemetry=telemetry,extensions=extensions)
            if extensions is not None and (telemetry.loss_due or telemetry.grad_due):
                for name,values in extensions.records.items():
                    telemetry.add_sum('optional/'+name,sum(values)/len(values))
                if hasattr(extensions,'diagnostic_history') and extensions.diagnostic_history:
                    latest=extensions.diagnostic_history[-1]
                    if latest['phase_step']==step:
                        telemetry.add_sum('optional/GAN_FKL_parameter_ratio',latest['weighted_parameter_ratio'])
                        telemetry.add_sum('optional/GAN_FKL_parameter_cosine',latest['parameter_cosine'])
            telemetry.publish(acc,out,losses,optimizers,time.monotonic()-started)
        except Exception as error:
            if isinstance(error,torch.cuda.OutOfMemoryError):
                print('OOM: configure micro_batch/resolution/precision/checkpointing explicitly; DDP replicates all four Transformers. No automatic fallback was applied.',flush=True)
            with (out/'logs'/f'failure-rank{acc.process_index}.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step+1,rank=acc.process_index,
                    error_type=type(error).__name__,error=str(error),
                    oom_diagnostic=('OOM: lower explicit micro_batch or resolution, increase accumulation to preserve batch, or use BF16/checkpointing; DDP replicates all four Transformers on each GPU. No settings were changed.' if isinstance(error,torch.cuda.OutOfMemoryError) else None),
                    nonfinite_detected=('nonfinite' in str(error).lower() or 'nan' in str(error).lower())))+'\n')
            raise
        step += 1
        if stream.consumed != step * microbatches_per_update(c):
            raise RuntimeError('DiMO data progress differs from completed two-stage updates')
        emit=checking or step % c.get('monitoring',{}).get('loss_every',10)==0 or step==c['max_steps']
        if emit:
            if extensions is not None:
                for key in extensions.records:
                    losses[key]=[float(v.detach()) if torch.is_tensor(v) else v for v in losses[key]]
            # One transfer for all microbatch losses at the logging boundary.
            scalars=torch.stack([*losses['student'],*[x for r in losses['auxiliary'] for x in r]]).float().cpu().tolist()
            n=len(losses['student']);width=c['gradient_accumulation']
            losses['student']=scalars[:n]
            losses['auxiliary']=[scalars[i:i+width] for i in range(n,len(scalars),width)]
            with (out/'logs'/f'rank{acc.process_index}.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step,losses=losses,consumed_batches=stream.consumed,
                    peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started))+'\n')
            acc.print(f'DiMO step {step}/{c["max_steps"]}: '
                      f'Student={sum(losses["student"])/len(losses["student"]):.6g}, '
                      f'Auxiliary={sum(losses["auxiliary"][0])/len(losses["auxiliary"][0]):.6g}, '
                      f'EMA={losses["ema_decay"]:.9f}, {time.monotonic()-started:.2f}s')
        preflight = step == (min(c['max_steps'],state['global_step']+c['preflight_steps']) if resume else c['preflight_steps'])
        if preflight or step % c.get('preview_every',500)==0:
            source,original,cond=probe['source'],probe['original'],probe['cond']
            # Save/recover RNG so monitoring samples cannot alter training.
            from src.dimo.checkpoint import rng_state,restore_rng
            before=rng_state()
            try:
                collective_call(acc,lambda:preview(acc,out,step,ema,vq,source,original,cond,
                    None,r_init=c['student_initial_mask_ratio']))
            finally:restore_rng(before)
        if preflight or step % c['save_every']==0 or step==c['max_steps']:
            checkpoint = save_checkpoint(acc, out, step, stream.consumed, c, models, optimizers, schedulers,extensions=extensions)
            if preflight:
                progress = check_roundtrip(acc, checkpoint, c, models, optimizers, schedulers,extensions=extensions)
                # Compare actual next data batches, then reconstruct at the saved offset.
                expected_next = state_digest(next(stream))
                replay = dataset_stream(c, tokenizer, acc, progress['consumed_batches'])
                if state_digest(next(replay)) != expected_next:
                    raise RuntimeError('Checkpoint data replay differs from uninterrupted stream')
                stream = dataset_stream(c, tokenizer, acc, progress['consumed_batches'])
                write_json(out / 'preflight' / f'passed-rank{acc.process_index}-step{step}.json',
                           dict(step=step, peak_bytes=torch.cuda.max_memory_allocated(), hardware=hardware,
                                checkpoint_reload=True, data_replay=True, gradient_isolation=True))
                acc.print(f"{c['preflight_steps']}-step GPU preflight and checkpoint reload passed")
                collective_call(acc,lambda:prune_checkpoints(out,c),main_only=True)
                if args.stop_after_preflight:
                    acc.wait_for_everyone()
                    break
            else:
                collective_call(acc,lambda:prune_checkpoints(out,c),main_only=True)
    acc.wait_for_everyone()
    acc.end_training()
    if output_lock is not None:
        output_lock.close()


if __name__ == '__main__':
    main()
