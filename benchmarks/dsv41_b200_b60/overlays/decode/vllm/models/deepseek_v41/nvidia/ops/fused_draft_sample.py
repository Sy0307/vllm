# SPDX-License-Identifier: Apache-2.0
"""Exact DSpark logits-add/Gumbel/cache/reduction fusion for NVIDIA.

Retains the original Markov GEMM and vLLM's random-number and temperature
helpers. Only fuses elementwise work and reductions, preserving rounding,
cache indexing, invalid padded rows and lowest-index tie breaking.
"""
import os
import torch
from vllm.triton_utils import triton, tl
from vllm.logger import init_logger
logger=init_logger(__name__)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed


@triton.jit
def _sample_blocks(base, bias, indices, positions, temperatures, seeds,
                   cache, maxima, arguments, base_stride:tl.constexpr,
                   bias_stride:tl.constexpr, idx_stride:tl.constexpr,
                   pos_stride:tl.constexpr, cache_stride0:tl.constexpr,
                   cache_stride1:tl.constexpr, VOCAB:tl.constexpr,
                   BLOCKS:tl.constexpr, STEP:tl.constexpr,
                   FP64:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64); b=tl.program_id(1)
    v=b*BLOCK+tl.arange(0,BLOCK); mask=v<VOCAB
    a=tl.load(base+row*base_stride+v,mask=mask,other=0.)
    z=tl.load(bias+row*bias_stride+v,mask=mask,other=0.)
    # Match the materialized torch.add output's rounding before Gumbel/cache.
    logits=(a.to(tl.float32)+z.to(tl.float32)).to(base.dtype.element_ty)
    req=tl.load(indices+row*idx_stride).to(tl.int64);valid=req>=0
    tl.store(cache+req*cache_stride0+STEP*cache_stride1+v,logits,mask=mask&valid)
    seed=tl.load(seeds+req,mask=valid,other=0)
    seed=spec_step_seed(seed,STEP)
    pos=tl.load(positions+row*pos_stride)-1
    temp=tl.load(temperatures+req,mask=valid,other=0.).to(tl.float32)
    # NaN-safe: an all-NaN row (garbage DP dummy rows) must not pick the first padded lane (token == VOCAB),
    # which the next DSpark step would feed to the Markov embedding (out-of-range index_select).
    logits=logits.to(tl.float32)
    logits=tl.where(mask&(logits==logits),logits,float('-inf'))
    val,idx=gumbel_noised_argmax(logits,v,mask,seed,pos,temp,
         IS_DRAFTING=True,USE_FP64=FP64,APPLY_TEMPERATURE=True)
    tl.store(maxima+row*BLOCKS+b,val)
    tl.store(arguments+row*BLOCKS+b,b*BLOCK+idx)


@triton.jit
def _reduce(maxima,arguments,out,out_stride:tl.constexpr,
            BLOCKS:tl.constexpr,WIDTH:tl.constexpr,STEP:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);j=tl.arange(0,WIDTH)
    vals=tl.load(maxima+row*BLOCKS+j,mask=j<BLOCKS,other=float('-inf'))
    winner=tl.argmax(vals,axis=0,tie_break_left=True)
    token=tl.load(arguments+row*BLOCKS+winner)
    tl.store(out+row*out_stride+STEP,token)


def sample_with_bias(base,bias,idx,positions,temperatures,seeds,cache,out,step,use_fp64=False):
    assert base.shape==bias.shape and base.dtype==bias.dtype==cache.dtype
    assert base.stride(-1)==bias.stride(-1)==cache.stride(-1)==1
    rows,vocab=base.shape;blocks=triton.cdiv(vocab,1024)
    maxima=torch.empty((rows,blocks),device=base.device,dtype=torch.float64 if use_fp64 else torch.float32)
    arguments=torch.empty((rows,blocks),device=base.device,dtype=torch.int64)
    _sample_blocks[(rows,blocks)](base,bias,idx,positions,temperatures,seeds,
        cache,maxima,arguments,base.stride(0),bias.stride(0),idx.stride(0),
        positions.stride(0),cache.stride(0),cache.stride(1),vocab,blocks,
        step,use_fp64,1024)
    _reduce[(rows,)](maxima,arguments,out,out.stride(0),blocks,
                    triton.next_power_of_2(blocks),step)
    return out[:rows,step]


def sample_sequential(spec,num_reqs,head_hidden):
    if os.environ.get('VLLM_DSV41_FUSED_DRAFT_SAMPLE','0')!='1':return False
    if (spec.draft_logits is None or spec.draft_watermarker is not None
        or spec._d2t_scatter_index is not None or spec.use_confidence_head
        or spec.enable_adaptive_verification):return False
    logger.info_once('OPT DSpark fused add/Gumbel/cache/reduction active')
    n=spec.num_speculative_steps;ns=num_reqs*n
    hidden=head_hidden[spec.sample_indices[:ns]]
    base=spec.model.compute_draft_logits(hidden).view(num_reqs,n,-1)
    idx=spec.sample_idx_mapping[:ns].view(num_reqs,n)
    pos=spec.sample_pos[:ns].view(num_reqs,n)
    prev=spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    for i in range(n):
        bias=spec.model.markov_bias(spec.model.markov_embed(prev))
        b=base[:,i]
        if b.dtype==bias.dtype==spec.draft_logits.dtype and b.stride(-1)==bias.stride(-1)==1:
            prev=sample_with_bias(b,bias,idx[:,i],pos[:,i],spec.temperature,
                spec.seeds,spec.draft_logits,spec.draft_tokens,i,spec.use_fp64_gumbel)
        else:
            prev=spec._sample_logits(b+bias,idx[:,i],pos[:,i],i)
            spec.draft_tokens[:num_reqs,i]=prev
    return True
