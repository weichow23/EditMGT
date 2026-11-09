"""Teacher-embedding regional logistic GAN; original Transformer stays unchanged."""
from contextlib import contextmanager
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.sparse_utils import prepare_sparse_target_layout


def generator_loss(fake_logits):
    return F.softplus(-fake_logits.float()).mean()


def discriminator_loss(real_logits, fake_logits):
    return .5 * (F.softplus(fake_logits.float()).mean() + F.softplus(-real_logits.float()).mean())


def phase_enabled(completed_phase_updates, warmup=200):
    if type(completed_phase_updates) is not int or completed_phase_updates < 0:
        raise ValueError('GAN phase counter is completed new-stage Student updates')
    return completed_phase_updates >= warmup


class GANRandom:
    """Independent checkpointed CUDA/CPU RNG; never consumes FKL sampling RNG."""
    def __init__(self, device, seed):
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def state_dict(self):
        return {'generator': self.generator.get_state().cpu()}

    def load_state_dict(self, state):
        self.generator.set_state(state['generator'].cpu())

    def sample(self, region):
        if region.dtype != torch.bool or region.ndim != 3:
            raise ValueError('G must be a boolean B,H,W generation mask')
        flat = region.flatten(1)
        count = flat.sum(1)
        if (count < 1).any():
            raise ValueError('GAN requires at least one active G token')
        ratio = torch.rand((len(region),), device=region.device, generator=self.generator) * .5
        # Ratios are drawn U(0,.5); retain at least one unmasked token even for
        # a one-token G. Unlike original FKL Q, GAN may mask zero tokens.
        masked_count = torch.minimum((count * ratio).floor().long(), count - 1)
        order = torch.rand(flat.shape, device=region.device, generator=self.generator).masked_fill(~flat, 2).argsort(1).argsort(1)
        mask = ((order < masked_count[:, None]) & flat).reshape_as(region)
        assert not (mask & ~region).any() and ((region & ~mask).flatten(1).sum(1) > 0).all()
        return ratio, mask


def expected_embedding(logits, source, region, teacher):
    """T=1 expectation over Teacher token lookup, never the VQ codebook."""
    if logits.ndim != 4 or logits.shape[:3] != source.shape or region.shape != source.shape:
        raise ValueError('Expected logits B,H,W,V and aligned Source/G')
    table = teacher.embed.embeddings.weight.detach()
    if table.requires_grad or logits.shape[-1] != teacher.config.codebook_size:
        raise ValueError('Teacher lookup/code vocabulary mismatch')
    source_embedding = F.embedding(source, table)
    # Only G logits participate, so their complement has an exactly zero
    # derivative, including when stem convolutions mix neighboring positions.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        selected = logits[region].float().softmax(-1) @ table[:logits.shape[-1]].float()
    return source_embedding.to(selected.dtype).masked_scatter(region[..., None].expand_as(source_embedding), selected.reshape(-1))


def real_embedding(target, source, region, teacher):
    table = teacher.embed.embeddings.weight.detach()
    if target.shape != source.shape or (target < 0).any() or (target >= teacher.config.codebook_size).any():
        raise ValueError('Invalid or unaligned Target VQ tokens')
    codes = torch.where(region, target, source)
    return F.embedding(codes, table)


def mask_embedding(embedding, mask, teacher):
    value = teacher.embed.embeddings.weight[teacher.config.vocab_size-1].detach().to(embedding.dtype)
    return torch.where(mask[..., None], value, embedding)


def restore_spatial(features, layout):
    """Restore actual row/column indices and a separate G/valid mask."""
    if features.shape[:2] != layout.indices.shape or (~layout.valid.any(1)).any():
        raise ValueError('Teacher feature sequence does not match active sparse layout')
    b, _, channels = features.shape
    n = math.prod(layout.grid_size)
    dense = layout.scatter(features, features.new_zeros(b, n, channels))
    active = layout.scatter(features.new_ones(b, features.shape[1], 1), features.new_zeros(b, n, 1))
    h, w = layout.grid_size
    return dense.transpose(1, 2).reshape(b, channels, h, w), active.transpose(1, 2).reshape(b, 1, h, w)


def restore_and_average(features, layout, roi_size=16):
    """Historical name; now returns spatial ROI plus validity, NEVER a mean.

    A mask-normalized bilinear resize keeps disconnected/irregular G cells
    in their original arrangement. Invalid cells and batch padding contribute
    no feature value or gradient. The last channel exposes valid coverage.
    """
    dense, valid = restore_spatial(features, layout)
    rois = []
    for index in range(len(dense)):
        positions = valid[index, 0].nonzero()
        y0, x0 = positions.min(0).values.tolist()
        y1, x1 = (positions.max(0).values + 1).tolist()
        crop = dense[index:index+1, :, y0:y1, x0:x1]
        support = valid[index:index+1, :, y0:y1, x0:x1]
        coverage = F.interpolate(support.float(), (roi_size, roi_size), mode='bilinear', align_corners=False)
        resized = F.interpolate((crop * support).float(), (roi_size, roi_size), mode='bilinear', align_corners=False)
        normalized = torch.where(coverage > 0, resized / coverage.clamp_min(1e-6), 0.)
        rois.append(torch.cat((normalized, coverage), 1).to(features.dtype))
    return torch.cat(rois, 0)


class _FeaturesComplete(Exception):
    pass


class TeacherFeatures:
    """Scoped original-forward hooks; no replacement of Teacher.forward.

    The GAN-only call stops after the last selected joint block, as in the
    official Meissonic feature discriminator. Reference/text/position/masks
    are still produced by the exact regional-v2 forward. Whole-call
    checkpointing reinstalls hooks during recomputation, rather than leaving
    removed hooks outside a backward checkpoint boundary.
    """
    def __init__(self, teacher, layers=(4, 7, 10, 13), activation_checkpointing=True, roi_size=16):
        if teacher.training or any(p.requires_grad for p in teacher.parameters()):
            raise ValueError('Teacher feature backbone must be frozen and eval')
        self.teacher = teacher
        self.layers = tuple(layers)
        self.activation_checkpointing = activation_checkpointing
        self.roi_size = roi_size
        if len(set(layers)) != 4 or min(layers) < 0 or max(layers) >= len(teacher.transformer_blocks):
            raise ValueError('Require four distinct valid joint-block feature depths')
        self.last_layout = None

    def __call__(self, embedding, source, original, conditions, ratio):
        teacher = self.teacher
        if not torch.equal(conditions['target_edit_mask'], original):
            raise ValueError('GAN must receive the original M, not G or FKL Q')
        h, w = source.shape[-2:]
        grid = (h//2, w//2) if teacher.down_block.downsample is not None else (h, w)
        layout = prepare_sparse_target_layout(original, (h,w), grid, len(source), source.device, teacher.config.sparse_target_halo)
        self.last_layout = layout
        def run(value):
            found, handles, lookups = {}, [], [0]
            def inject(module, inputs, output):
                lookups[0] += 1
                # Original v2 calls target lookup first, reference second.
                if lookups[0] == 1:
                    if output.shape != value.shape:raise ValueError('Teacher raw lookup embedding shape mismatch')
                    return value.to(output.dtype)
                return output
            handles.append(teacher.embed.embeddings.register_forward_hook(inject))
            def feature_hook(index):
                def capture(module, inputs, output):
                    if not isinstance(output, tuple) or len(output) != 3:
                        raise ValueError('Expected joint Text/Target/Reference block outputs')
                    found[index] = restore_and_average(output[1], layout, self.roi_size)
                    if index == max(self.layers):raise _FeaturesComplete()
                return capture
            for index in self.layers:
                handles.append(teacher.transformer_blocks[index].register_forward_hook(feature_hook(index)))
            try:
                teacher(hidden_states=source, timestep=ratio, mask_ratio=ratio, sparse_target=True, **conditions)
                raise RuntimeError('GAN feature stop was not reached')
            except _FeaturesComplete:
                if lookups[0] != 2 or set(found) != set(self.layers):
                    raise RuntimeError('Teacher feature capture incomplete or lookup order changed')
                return tuple(found[index] for index in self.layers)
            finally:
                for handle in handles:handle.remove()
        if self.activation_checkpointing and embedding.requires_grad and torch.is_grad_enabled():
            return checkpoint(run, embedding, use_reentrant=False)
        return run(embedding)


class SpatialHead(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        groups = math.gcd(8, hidden)
        small = hidden // 2
        self.project = nn.Conv2d(channels + 1, hidden, 1)
        self.convs = nn.Sequential(
            nn.GroupNorm(groups, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(groups, hidden), nn.SiLU(),
            nn.Conv2d(hidden, small, 3, stride=2, padding=1),
            nn.GroupNorm(math.gcd(8, small), small), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)))

    def forward(self, features, time):
        return self.convs(self.project(features) + time[:, :, None, None]).flatten(1)


class RegionalDiscriminator(nn.Module):
    """Four spatial ROI Conv/GN/SiLU heads; Teacher is never registered here."""
    def __init__(self, channels, hidden=128, num_layers=4):
        super().__init__()
        self.time_dim = channels//4
        if self.time_dim < 4 or self.time_dim % 2:
            raise ValueError('Teacher dimension must support even sinusoidal time embedding')
        self.time_projections = nn.ModuleList(nn.Linear(self.time_dim, hidden) for _ in range(num_layers))
        self.heads = nn.ModuleList(SpatialHead(channels, hidden) for _ in range(num_layers))
        self.final = nn.Linear((hidden//2)*4*4*num_layers, 1)

    def forward(self, features, ratio):
        if len(features) != len(self.heads) or ratio.shape != (len(features[0]),):
            raise ValueError('Discriminator feature/time shape mismatch')
        frequencies = 10000 ** (-torch.arange(self.time_dim//2, device=ratio.device, dtype=torch.float32)/(self.time_dim//2-1))
        angle = (ratio.float()*1000)[:,None]*frequencies[None]
        time = torch.cat((angle.sin(), angle.cos()), dim=-1)
        if any(feature.ndim != 4 for feature in features):
            raise ValueError('Spatial discriminator requires BCHW ROI features with validity channel')
        outputs = [head(feature, projection(time.to(feature.dtype))) for feature,projection,head in zip(features,self.time_projections,self.heads)]
        return self.final(torch.cat(outputs,dim=-1)).squeeze(-1)


@contextmanager
def frozen_discriminator(discriminator):
    bare = getattr(discriminator, 'module', discriminator)
    flags = [p.requires_grad for p in bare.parameters()]
    mode = bare.training
    try:
        bare.eval().requires_grad_(False)
        yield bare
    finally:
        for parameter, flag in zip(bare.parameters(), flags):parameter.requires_grad_(flag)
        bare.train(mode)


def adversarial_generator(logits, source, original, region, conditions, teacher, backbone, discriminator, random):
    ratio, mask = random.sample(region)
    embedding = mask_embedding(expected_embedding(logits,source,region,teacher),mask,teacher)
    features = backbone(embedding,source,original,conditions,ratio)
    with frozen_discriminator(discriminator) as bare:
        score = bare(features,ratio)
    return generator_loss(score), score, ratio, mask
