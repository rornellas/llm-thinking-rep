import torch
import pytest
from pre_qwen_certification.modal import MoEGeometry
from pre_qwen_certification.heterogeneous_rank import HeterogeneousSharedLowRankResidualMoE
from pre_qwen_certification.vectorized_compact import VectorizedCompactMoE

@pytest.mark.parametrize('rank',[1,8])
@pytest.mark.parametrize('geometry',[MoEGeometry(32,64,12,4),MoEGeometry(16,24,5,2)])
def test_values_forward_and_gradients(rank,geometry):
    torch.set_num_threads(2)
    torch.manual_seed(904201)
    a=HeterogeneousSharedLowRankResidualMoE(geometry,(rank,)*geometry.n_experts)
    # Non-tiny factors exercise nonlinear and residual gradients, not only common paths.
    with torch.no_grad():
        for bank in ('gate','up','down'):
            for side in ('left','right'):
                for p in getattr(a,bank+'_'+side):p.mul_(10)
    b=VectorizedCompactMoE(a)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for bank in ('gate','up','down'):
        assert torch.equal(getattr(a,'common_'+bank),getattr(b,'common_'+bank))
        for side in ('left','right'):
            assert torch.equal(torch.stack(list(getattr(a,bank+'_'+side))),getattr(b,bank+'_'+side))
    x=torch.randn(19,geometry.d_model,requires_grad=True)
    y=x.detach().clone().requires_grad_(True)
    u,ru=a(x);v,rv=b(y)
    assert torch.equal(ru.top_ids,rv.top_ids)
    assert torch.allclose(u,v,atol=2e-5,rtol=2e-4)
    target=torch.randn_like(u)
    (u*target).sum().backward();(v*target).sum().backward()
    assert torch.allclose(x.grad,y.grad,atol=3e-5,rtol=3e-4)
    assert torch.allclose(a.router.weight.grad,b.router.weight.grad,atol=3e-5,rtol=3e-4)
    for bank in ('gate','up','down'):
        assert torch.allclose(getattr(a,'common_'+bank).grad,getattr(b,'common_'+bank).grad,atol=3e-5,rtol=3e-4)
        for side in ('left','right'):
            expected=torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in getattr(a,bank+'_'+side)])
            assert torch.allclose(expected,getattr(b,bank+'_'+side).grad,atol=3e-5,rtol=3e-4)


def test_nonuniform_rank_rejected():
    a=HeterogeneousSharedLowRankResidualMoE(MoEGeometry(16,24,5,2),(1,2,1,2,1))
    with pytest.raises(ValueError):VectorizedCompactMoE(a)
