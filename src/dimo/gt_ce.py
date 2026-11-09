"""Optional original-M token supervision; independent of the v3 FKL loss."""
import math
import torch
import torch.nn.functional as F


def validate_weight(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('gt_ce_weight must be finite and nonnegative')
    return float(value)


def region_token_ce(logits, target, original_mask):
    """Mean over original editable tokens per sample, then sample mean.

    logits: [B,H,W,V]; labels/mask: [B,H,W]. Neither Q nor Halo
    participates in selecting the CE positions. Keep the original vocabulary.
    """
    if logits.shape[:-1] != target.shape or target.shape != original_mask.shape:
        raise ValueError('GT tokens, original mask and Student logits must align')
    if target.dtype != torch.long or original_mask.dtype != torch.bool:
        raise ValueError('Expected long GT tokens and boolean original mask')
    counts = original_mask.flatten(1).sum(1)
    if (counts == 0).any():
        raise ValueError('Empty original edit mask in GT CE')
    selected = target[original_mask]
    if (selected < 0).any() or (selected >= logits.shape[-1]).any():
        raise ValueError('GT token outside Student vocabulary')
    # Selecting first avoids allocating full-grid FP32 cross entropy activations.
    sums = []
    for z, t, m in zip(logits, target, original_mask):
        sums.append(F.cross_entropy(z[m].float(), t[m], reduction='mean'))
    return torch.stack(sums).mean()


@torch.no_grad()
def encode_target(batch, vq, source, device):
    from src.v2_utils import prepare_cond_token
    if vq.training or any(p.requires_grad for p in vq.parameters()):
        raise ValueError('GT encoder must be frozen and in eval mode')
    image = batch['image'].to(device)
    if image.shape != batch['reference_image'].shape:
        raise ValueError('Source and target image tensors must have identical shape')
    # Match prepare_batch's FP32 VQ path even when CE is added inside the
    # mixed-precision Student forward. Autocast must not change target IDs.
    with torch.autocast(device_type=source.device.type, enabled=False):
        target = prepare_cond_token(None, image, vq).reshape_as(source)
    if target.dtype != torch.long or (target < 0).any() or (target >= vq.config.num_vq_embeddings).any():
        raise ValueError('Invalid target VQ tokens')
    return target
