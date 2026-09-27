"""Chunk-permute copies for the MLA head-group fold (replaces generic strided copies).

fold:   x [B, S, G*hg, D] (contiguous)  -> y [B*G, S, hg, D]   (y[b*G+g, s] = x[b, s, g*hg:(g+1)*hg])
unfold: y [B*G, S, hg, L] (contiguous)  -> x [B*S, G*hg, L]    (inverse)
Both move contiguous chunks of hg*D elements; one program per chunk, vectorized over
the chunk viewed as int32 words (chunk bytes must be a multiple of 4)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_perm_kernel(src, dst, S, G, W: tl.constexpr, BLOCK: tl.constexpr, FOLD: tl.constexpr):
    pid = tl.program_id(0)  # chunk index in (b, s, g) order of the [B, S, G] source layout (FOLD) / dst layout (unfold)
    b = pid // (S * G)
    r = pid % (S * G)
    s = r // G
    g = r % G
    a = pid.to(tl.int64) * W                    # offset in [B, S, G, W]
    c = ((b * G + g) * S + s).to(tl.int64) * W  # offset in [B, G, S, W]
    if FOLD:
        src_off, dst_off = a, c
    else:
        src_off, dst_off = c, a
    for i in range(0, W, BLOCK):
        o = i + tl.arange(0, BLOCK)
        m = o < W
        tl.store(dst + dst_off + o, tl.load(src + src_off + o, mask=m), mask=m)


def _words(t):
    return t.view(torch.int32) if t.element_size() * t.shape[-1] % 4 == 0 else None


def fold_heads(x, G, out=None):
    B, S, H, D = x.shape
    hg = H // G
    if out is None:
        out = torch.empty(B * G, S, hg, D, dtype=x.dtype, device=x.device)
    xs, ys = x.contiguous().view(-1).view(torch.int32), out.view(-1).view(torch.int32)
    W = hg * D * x.element_size() // 4
    _chunk_perm_kernel[(B * S * G,)](xs, ys, S, G, W=W, BLOCK=1024, FOLD=True, num_warps=4)
    return out


def unfold_heads(y, B, G, out=None):
    BG, S, hg, L = y.shape
    if out is None:
        out = torch.empty(B * S, G * hg, L, dtype=y.dtype, device=y.device)
    ys, xs = y.contiguous().view(-1).view(torch.int32), out.view(-1).view(torch.int32)
    W = hg * L * y.element_size() // 4
    _chunk_perm_kernel[(B * S * G,)](ys, xs, S, G, W=W, BLOCK=1024, FOLD=False, num_warps=4)
    return out
