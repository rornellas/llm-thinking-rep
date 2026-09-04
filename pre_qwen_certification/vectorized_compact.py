"""Function-preserving vectorized inference for uniform shared-residual MoE.

This is an implementation variant, not a new parameterization or trained model.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
import torch.nn.functional as F
from .heterogeneous_rank import HeterogeneousSharedLowRankResidualMoE
from .modal import Routing, route_topk, _validate_forced_routing


class VectorizedCompactMoE(nn.Module):
    def __init__(self, source: HeterogeneousSharedLowRankResidualMoE):
        super().__init__()
        if not isinstance(source,HeterogeneousSharedLowRankResidualMoE):
            raise TypeError('source must be a shared low-rank residual MoE')
        if len(set(source.ranks))!=1:
            raise ValueError('only uniform rank is supported')
        self.geometry=source.geometry
        self.rank=int(source.ranks[0])
        self.ranks=source.ranks
        self.router=copy.deepcopy(source.router)
        for bank in ('gate','up','down'):
            common=getattr(source,'common_'+bank)
            setattr(self,'common_'+bank,nn.Parameter(common.detach().clone()))
            for side in ('left','right'):
                values=torch.stack([p.detach() for p in getattr(source,bank+'_'+side)]).contiguous()
                setattr(self,bank+'_'+side,nn.Parameter(values))
        self.train(source.training)

    def forward(self,x:torch.Tensor,forced_top_ids=None,forced_weights=None):
        if x.ndim!=2 or x.shape[-1]!=self.geometry.d_model:
            raise ValueError('expected [tokens, d_model]')
        _validate_forced_routing(x,forced_top_ids,forced_weights,self.geometry.top_k)
        route=route_topk(x,self.router.weight,self.geometry.top_k)
        ids=route.top_ids if forced_top_ids is None else forced_top_ids
        weights=route.weights if forced_weights is None else forced_weights
        gate_coeff=torch.einsum('nkrd,nd->nkr',self.gate_right[ids],x)
        up_coeff=torch.einsum('nkrd,nd->nkr',self.up_right[ids],x)
        gate=F.linear(x,self.common_gate)[:,None,:]+torch.einsum('nkhr,nkr->nkh',self.gate_left[ids],gate_coeff)
        up=F.linear(x,self.common_up)[:,None,:]+torch.einsum('nkhr,nkr->nkh',self.up_left[ids],up_coeff)
        hidden=F.silu(gate)*up
        shared=F.linear(torch.sum(weights[:,:,None]*hidden,dim=1),self.common_down)
        down_coeff=torch.einsum('nkrh,nkh->nkr',self.down_right[ids],hidden)
        residual=torch.einsum('nkdr,nkr->nkd',self.down_left[ids],down_coeff)
        return shared+torch.sum(weights[:,:,None]*residual,dim=1),Routing(route.logits,ids,weights)

    def expert_transform_parameter_count(self):
        g=self.geometry
        return 3*g.d_ff*g.d_model+3*(g.d_ff+g.d_model)*g.n_experts*self.rank


def vectorize_inference_model(model):
    """Clone a model and replace shared-residual modules only; never keep originals."""
    result=copy.deepcopy(model)
    for block in result.blocks:
        block.moe=VectorizedCompactMoE(block.moe)
    return result
