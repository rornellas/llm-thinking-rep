"""Synthetic preflight, no scientific checkpoint access."""
import copy
import torch
from pre_qwen_certification.functional_ablation import InferenceLM, transform, accounting
from pre_qwen_certification.native_compact import build_paired_candidate_models, NativeArchitectureSpec, PRIMARY
from pre_qwen_certification.tiny_lm import TinyLMConfig
from pre_qwen_certification.modal import ConventionalSwiGLUMoE

def models():
    torch.set_num_threads(2)
    return build_paired_candidate_models(64, TinyLMConfig(n_experts=12), NativeArchitectureSpec(8,42), seed=19087)

@torch.no_grad()
def test_original_wrapper_parity():
    x = torch.arange(32).reshape(2,16)
    for m in models().values():
        m.eval()
        y = InferenceLM(m,'original')(x)
        assert torch.allclose(y,m(x)[0],rtol=1e-5,atol=1e-6)
        assert accounting(InferenceLM(m,'original'))['total_parameters']==sum(p.numel() for p in m.parameters())

@torch.no_grad()
def test_zero_residual_common_and_export_removal():
    m = models()[PRIMARY]
    for b in m.blocks:
        for bank in ('gate','up','down'):
            for left in getattr(b.moe,bank+'_left'):
                left.zero_()
    x = torch.arange(16)[None,:]
    common = InferenceLM(m,'common-only')
    assert torch.allclose(common(x),InferenceLM(m,'original')(x),rtol=1e-5,atol=1e-6)
    assert accounting(common)['router_parameters']==0
    assert all('router' not in name and '_left' not in name and '_right' not in name for name,_ in common.named_parameters())
    assert accounting(common)['total_parameters'] < accounting(InferenceLM(m,'original'))['total_parameters']

@torch.no_grad()
def test_rank1_exactness():
    m = models()[PRIMARY].blocks[0].moe
    for bank in ('gate','up','down'):
        for left,right in zip(getattr(m,bank+'_left'),getattr(m,bank+'_right'),strict=True):
            left[:,1:].zero_(); right[1:,:].zero_()
    truncated = transform(m,'rank1')
    x = torch.randn(17,32)
    assert torch.allclose(m(x)[0],truncated(x)[0],rtol=2e-5,atol=2e-6)
    assert all(p.shape[1]==1 for p in truncated.gate_left)

@torch.no_grad()
def test_permutation_router_reindex_invariance():
    # Route permutation alone changes semantics; bank + router reindex preserves it.
    x = torch.randn(13,32)
    for model in models().values():
        m = model.blocks[0].moe
        changed = copy.deepcopy(m)
        perm = (torch.arange(12)+5)%12
        changed.router.weight.copy_(m.router.weight[perm])
        if isinstance(m,ConventionalSwiGLUMoE):
            for bank in ('gate','up','down'):
                getattr(changed,bank).copy_(getattr(m,bank)[perm])
        else:
            for bank in ('gate','up','down'):
                for suffix in ('left','right'):
                    for i in range(12):
                        getattr(changed,bank+'_'+suffix)[i].copy_(getattr(m,bank+'_'+suffix)[int(perm[i])])
        assert torch.allclose(m(x)[0],changed(x)[0],rtol=2e-5,atol=2e-6)

@torch.no_grad()
def test_mean_is_mean_matrices_and_not_unused_banks():
    m = models()['conventional-full'].blocks[0].moe
    dense = transform(m,'mean-matrices')
    assert torch.allclose(dense.gate,m.gate.mean(0))
    assert sum(p.numel() for p in dense.parameters())==3*32*64

@torch.no_grad()
def test_route_interventions_definition():
    m = models()['conventional-full'].blocks[0].moe
    x = torch.randn(11,32)
    _, r = m(x)
    for kind in ('permute-1','permute-5','permute-7','uniform-selected'):
        expected_ids = r.top_ids if kind=='uniform-selected' else (r.top_ids+int(kind.split('-')[1]))%12
        expected_weights = torch.full_like(r.weights,.25) if kind=='uniform-selected' else r.weights
        reference = m(x,forced_top_ids=expected_ids,forced_weights=expected_weights)[0]
        assert torch.allclose(reference,transform(m,kind)(x)[0],rtol=1e-5,atol=1e-6)
