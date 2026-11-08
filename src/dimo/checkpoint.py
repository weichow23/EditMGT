"""Complete DiMO state, rank-local RNG/data progress, and atomic publication."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from src.dimo.origin import sha256

CHECKPOINT_KIND = 'region-dimo-v3'
UPDATE_RULE = 'student_then_auxiliary_new_batches'


def microbatches_per_update(config):
    accumulation = config.get('gradient_accumulation', 1)
    fake_rounds = config.get('fake_rounds', 1)
    if any(type(value) is not int or value < 1 for value in (accumulation, fake_rounds)):
        raise ValueError('gradient_accumulation and fake_rounds must be positive integers')
    return accumulation * (1 + fake_rounds)


def validate_progress(progress, config, world_size, step=None):
    if progress.get('kind') != CHECKPOINT_KIND or progress.get('update_rule') != UPDATE_RULE:
        raise ValueError('Incompatible DiMO training state: v3 state required; historical v1/v2 cannot resume')
    saved = progress['config']
    # Constant LR has no budget-dependent schedule: extending only max_steps
    # preserves optimizer, data and RNG trajectories. Other changes stay strict.
    saved_core = {k: v for k, v in saved.items() if k != 'max_steps'}
    current_core = {k: v for k, v in config.items() if k != 'max_steps'}
    if saved_core != current_core or progress['world_size'] != world_size:
        raise ValueError('Resume requires identical config, data/batch layout and world size')
    if config.get('max_steps', 0) < saved.get('max_steps', 0):
        raise ValueError('Resume may extend max_steps, but cannot shorten the saved budget')
    per_update = microbatches_per_update(config)
    global_step = progress['global_step']
    if (type(global_step) is not int or global_step < 0
            or (step is not None and global_step != step)
            or progress.get('microbatches_per_update') != per_update
            or progress['consumed_batches'] != global_step * per_update):
        raise ValueError('Incompatible two-stage data progress: consumed_batches must include both phases')


def collective_call(acc, action, main_only=False):
    """Propagate I/O/preflight failures to every rank before the next collective."""
    result, error = None, None
    try:
        if not main_only or acc.is_main_process:
            result = action()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = [error]
    if acc.num_processes > 1:
        errors = [None] * acc.num_processes
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f'DiMO operation failed by rank: {errors}')
    return result


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda'] is not None:
        # Each DDP rank owns one local device; do not initialize other GPUs' RNGs.
        torch.cuda.set_rng_state(state['cuda'].cpu())


def training_state(acc, models, optimizers, schedulers, extensions=None):
    state=dict(kind=CHECKPOINT_KIND, update_rule=UPDATE_RULE,
                models={k: acc.unwrap_model(v).state_dict() for k, v in models.items()},
                optimizers={k: v.state_dict() for k, v in optimizers.items()},
                schedulers={k: v.state_dict() for k, v in schedulers.items()})
    if extensions is not None:state['loss_state']=extensions.state_dict()
    return state


def state_digest(value):
    """Streaming CPU hash: no full-model clone on GPU during the memory preflight."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().contiguous().cpu()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def save_checkpoint(acc, out, step, consumed, config, models, optimizers, schedulers, extensions=None):
    out = Path(out)
    path, temporary = out / f'checkpoint-{step}', out / f'.checkpoint-{step}.incomplete'
    progress = dict(kind=CHECKPOINT_KIND, update_rule=UPDATE_RULE,
                    global_step=step, consumed_batches=consumed,
                    microbatches_per_update=microbatches_per_update(config),
                    world_size=acc.num_processes, config=config, rng=rng_state())
    if extensions is not None:
        progress['loss_rank_state']=extensions.rank_state()
        extensions.validate_state(extensions.state_dict(),progress['loss_rank_state'],step)
    collective_call(acc, lambda: validate_progress(progress, config, acc.num_processes, step))

    def begin():
        if path.exists() or temporary.exists():
            raise FileExistsError(f'Refusing to overwrite checkpoint: {path} / {temporary}')
        temporary.mkdir(parents=True)
    collective_call(acc, begin, main_only=True)
    # Main owns common state; every rank owns its RNG/progress. No SFT optimizer is read.
    collective_call(acc, lambda: torch.save(training_state(acc, models, optimizers, schedulers,extensions),
                                          temporary / 'training.pt'), main_only=True)
    collective_call(acc, lambda: torch.save(progress, temporary / f'rank-{acc.process_index}.pt'))

    def publish():
        checksums = {p.name: sha256(p) for p in temporary.glob('*.pt')}
        required={'training.pt',*(f'rank-{rank}.pt' for rank in range(acc.num_processes))}
        if set(checksums)!=required:raise ValueError('Missing rank checkpoint files before publication')
        header=dict(kind=CHECKPOINT_KIND,update_rule=UPDATE_RULE,global_step=step,
                    world_size=acc.num_processes,checksums=checksums,config=config,
                    output_dir=str(out.resolve()),metadata=config.get('source_metadata',{}))
        (temporary/'COMPLETE').write_text(json.dumps(header,indent=2),encoding='utf-8')
        verify_checkpoint(temporary)
        temporary.rename(path)
    collective_call(acc, publish, main_only=True)
    return path


def prune_checkpoints(out, config):
    """Only published checkpoints owned by this exact new run, after save succeeds."""
    import re, shutil
    out=Path(out).resolve()
    if not config.get('output_dir') or Path(config['output_dir']).resolve()!=out:
        raise ValueError('Retention requires this run explicit output_dir')
    candidates=[]
    for p in out.glob('checkpoint-*'):
        m=re.fullmatch(r'checkpoint-(\d+)',p.name)
        if not m or p.is_symlink() or not p.is_dir() or not (p/'COMPLETE').is_file():continue
        header=json.loads((p/'COMPLETE').read_text())
        if header.get('kind')!=CHECKPOINT_KIND or header.get('output_dir')!=str(out):continue
        # Verify ownership/config including Teacher/source fingerprint before deletion.
        saved={k:v for k,v in header.get('config',{}).items() if k!='max_steps'}
        current={k:v for k,v in config.items() if k!='max_steps'}
        if saved!=current:continue
        candidates.append((int(m[1]),p))
    candidates.sort()
    limit=config.get('checkpoints_total_limit',5)
    keep={p for _,p in candidates[-limit:]}
    milestones=set(config.get('checkpoint_milestones',[]))
    deleted=[]
    for step,p in candidates:
        if p not in keep and step not in milestones:
            shutil.rmtree(p);deleted.append(str(p))
    return deleted


def verify_checkpoint(path):
    path = Path(path)
    manifest = json.loads((path / 'COMPLETE').read_text(encoding='utf-8'))
    if manifest.get('kind') != CHECKPOINT_KIND or manifest.get('update_rule') != UPDATE_RULE:
        raise ValueError('Incompatible DiMO checkpoint: v3 state required; historical v1/v2 cannot resume')
    required = {'training.pt', *(f'rank-{rank}.pt' for rank in range(manifest['world_size']))}
    if set(manifest['checksums']) != required:
        raise ValueError('Missing common training state or rank RNG/data progress')
    for name, digest in manifest['checksums'].items():
        if Path(name).name != name or sha256(path / name) != digest:
            raise ValueError(f'Incomplete/corrupt DiMO checkpoint file: {name}')
    return manifest


def read_progress(path, rank):
    # These files belong to this explicitly selected local DiMO output.
    return torch.load(Path(path) / f'rank-{rank}.pt', map_location='cpu', weights_only=False)


def load_checkpoint(acc, path, config, models, optimizers, schedulers, extensions=None):
    def load():
        manifest = verify_checkpoint(path)
        progress = read_progress(path, acc.process_index)
        validate_progress(progress, config, acc.num_processes, manifest['global_step'])
        if manifest['world_size'] != acc.num_processes:
            raise ValueError('Resume requires identical world size')
        state = torch.load(Path(path) / 'training.pt', map_location='cpu', weights_only=False)
        if state.get('kind') != CHECKPOINT_KIND or state.get('update_rule') != UPDATE_RULE:
            raise ValueError('Incompatible DiMO optimizer/model state: v3 state required')
        if set(state['models'])!=set(models) or set(state['optimizers'])!=set(optimizers) or set(state['schedulers'])!=set(schedulers):
            raise ValueError('Incompatible model/optimizer/scheduler roles')
        if extensions is not None:
            extensions.validate_state(state.get('loss_state',{}),progress.get('loss_rank_state',{}),progress['global_step'])
        elif 'loss_state' in state:raise ValueError('Optional loss checkpoint requires its configured runtime')
        for key, model in models.items():
            acc.unwrap_model(model).load_state_dict(state['models'][key], strict=True)
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(state['optimizers'][key])
        for key, scheduler in schedulers.items():
            scheduler.load_state_dict(state['schedulers'][key])
        if extensions is not None:extensions.load_state(state['loss_state'],progress['loss_rank_state'],progress['global_step'])
        restore_rng(progress['rng'])
        return progress
    return collective_call(acc, load)


def check_roundtrip(acc, path, config, models, optimizers, schedulers, extensions=None):
    before = state_digest(training_state(acc, models, optimizers, schedulers,extensions))
    rng_before = state_digest((rng_state(),extensions.rank_state() if extensions is not None else None))
    progress = load_checkpoint(acc, path, config, models, optimizers, schedulers,extensions)
    def compare():
        if before != state_digest(training_state(acc, models, optimizers, schedulers,extensions)):
            raise RuntimeError('Checkpoint reload changed Student/Auxiliary/EMA/optimizer/scheduler')
        if rng_before != state_digest((rng_state(),extensions.rank_state() if extensions is not None else None)):
            raise RuntimeError('Checkpoint RNG reload differs')
    collective_call(acc, compare)
    return progress
