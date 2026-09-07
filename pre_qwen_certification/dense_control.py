"""DCG-1 fixed recipe, dense baselines, and inference without auxiliary work."""
from __future__ import annotations
import copy
import hashlib
import torch
from torch import nn
import torch.nn.functional as F
from .tiny_lm import TinyLMConfig, TinyMoELanguageModel
from .heterogeneous_rank import HeterogeneousSharedLowRankResidualMoE
from .functional_ablation import DenseSwiGLU, transform, accounting
from .native_compact import stable_seed

SEEDS = (906101, 906103, 906107, 906109, 906119, 906121)
TRAIN_ARMS = ('native-rank1', 'native-rank8', 'dense104', 'dense80', 'dense164')
EVAL_ARMS = TRAIN_ARMS + ('rank8-to1',)
MILESTONES = (800, 2200, 4400)
CONFIG = TinyLMConfig(batch_size=8, n_experts=12, teacher_steps=4400,
                      student_steps=0, learning_rate=0.0003)


class ControlBlock(nn.Module):
    def __init__(self, source: nn.Module, moe: nn.Module):
        super().__init__()
        self.norm1 = copy.deepcopy(source.norm1)
        self.attention = copy.deepcopy(source.attention)
        self.norm2 = copy.deepcopy(source.norm2)
        self.moe = moe

    def forward(self, x: torch.Tensor, mask: torch.Tensor, auxiliary: bool):
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     attn_mask=mask, need_weights=False)
        x = x + attended
        output, routing = self.moe(self.norm2(x).flatten(0, 1))
        penalty = x.new_zeros(())
        if auxiliary and routing is not None:
            probabilities = F.softmax(routing.logits, dim=-1)
            importance = probabilities.mean(0)
            assignments = F.one_hot(routing.top_ids, probabilities.shape[-1]).float().mean((0, 1))
            balance = probabilities.shape[-1] * torch.sum(importance * assignments)
            penalty = balance + 0.1 * torch.logsumexp(routing.logits, -1).square().mean()
        return x + output.reshape_as(x), penalty


class ControlLM(nn.Module):
    def __init__(self, source: TinyMoELanguageModel, modules: list[nn.Module]):
        super().__init__()
        self.config = source.config
        self.token_embedding = copy.deepcopy(source.token_embedding)
        self.position_embedding = copy.deepcopy(source.position_embedding)
        self.blocks = nn.ModuleList(ControlBlock(b, m) for b, m in zip(source.blocks, modules, strict=True))
        self.norm = copy.deepcopy(source.norm)
        self.output = nn.Linear(self.config.d_model, source.token_embedding.num_embeddings, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor, *, auxiliary: bool = False):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.seq_len:
            raise ValueError('expected batch x sequence within the configured context')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        mask = torch.triu(x.new_full((tokens.shape[1], tokens.shape[1]), -float('inf')), diagonal=1)
        penalties = []
        for block in self.blocks:
            x, penalty = block(x, mask, auxiliary)
            penalties.append(penalty)
        return self.output(self.norm(x)), torch.stack(penalties).mean()


def build_models(seed: int, vocab: int = 512) -> dict[str, ControlLM]:
    torch.manual_seed(seed)
    base = TinyMoELanguageModel(vocab, CONFIG)
    models = {}
    for arm in TRAIN_ARMS:
        modules = []
        for layer in range(CONFIG.n_layers):
            torch.manual_seed(stable_seed(seed, 'compact' if arm.startswith('native') else arm, layer))
            if arm.startswith('native'):
                rank = 1 if arm == 'native-rank1' else 8
                module = HeterogeneousSharedLowRankResidualMoE(CONFIG.geometry, (rank,) * CONFIG.n_experts)
                with torch.no_grad():
                    module.router.weight.copy_(base.blocks[layer].moe.router.weight)
            else:
                width = int(arm.removeprefix('dense'))
                matrices = [torch.empty(width, CONFIG.d_model), torch.empty(width, CONFIG.d_model),
                            torch.empty(CONFIG.d_model, width)]
                for matrix in matrices:
                    nn.init.xavier_uniform_(matrix)
                module = DenseSwiGLU(*matrices)
            modules.append(module)
        models[arm] = ControlLM(base, modules)
    return models


def compress(model: ControlLM) -> ControlLM:
    result = copy.deepcopy(model)
    for block in result.blocks:
        block.moe = transform(block.moe, 'rank1')
    return result


def counts(model: ControlLM) -> dict[str, int]:
    result = accounting(model)
    result['ffn_matrix_macs_including_router'] = result['expert_matrix_macs_per_token'] + result['router_parameters']
    return result


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def self_test() -> dict:
    models = build_models(17)
    reference = models['native-rank1'].state_dict()
    expected = {'native-rank1': (47168, 15360), 'native-rank8': (95552, 31488),
                'dense104': (47168, 19968), 'dense80': (42560, 15360), 'dense164': (58688, 31488)}
    for name, model in models.items():
        actual = counts(model)
        assert (actual['total_parameters'], actual['ffn_matrix_macs_including_router']) == expected[name]
        assert model.output.weight is model.token_embedding.weight
        for key, value in model.state_dict().items():
            if '.moe.' not in key or '.moe.router.' in key:
                assert torch.equal(value, reference[key]), (name, key)
    for layer in range(CONFIG.n_layers):
        for bank in ('gate', 'up', 'down'):
            assert torch.equal(getattr(models['native-rank1'].blocks[layer].moe, 'common_' + bank),
                               getattr(models['native-rank8'].blocks[layer].moe, 'common_' + bank))
    again = build_models(17)
    assert all(state_hash(models[name]) == state_hash(again[name]) for name in TRAIN_ARMS)
    tokens = torch.arange(64).reshape(2, 32).long()
    errors = {}
    for name, model in models.items():
        model.train()
        logits, aux = model(tokens, auxiliary=True)
        serving, _ = model(tokens)
        assert torch.equal(logits, serving)
        if name.startswith('dense'):
            assert float(aux) == 0
            x = torch.randn(4, 32)
            m = model.blocks[0].moe
            manual = (F.silu(x @ m.gate.T) * (x @ m.up.T)) @ m.down.T
            assert torch.allclose(manual, m(x)[0], atol=1e-6, rtol=1e-5)
        else:
            old = TinyMoELanguageModel(512, CONFIG)
            for block in old.blocks:
                block.moe = HeterogeneousSharedLowRankResidualMoE(CONFIG.geometry, (1 if name.endswith('1') else 8,) * 12)
            old.load_state_dict(model.state_dict())
            previous, previous_aux, _ = old(tokens)
            errors[name] = float((previous - logits).abs().max().detach())
            assert torch.equal(previous, logits) and torch.equal(previous_aux, aux)
        loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten()) + .01 * aux
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        before = state_hash(model)
        torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.02).step()
        assert state_hash(model) != before
    reduced = compress(models['native-rank8'])
    assert counts(reduced)['total_parameters'] == 47168
    assert all(block.moe.ranks == (1,) * 12 for block in reduced.blocks)
    for block in reduced.blocks:
        for left, right in zip(block.moe.gate_left, block.moe.gate_right, strict=True):
            singular = torch.linalg.svdvals(left.detach().double() @ right.detach().double())
            assert float(singular[1] / singular[0]) < 1e-10
    return {'passed': True, 'parameter_and_compute_counts': expected,
            'legacy_training_forward_max_error': errors, 'checks': [
                'matched_backbone', 'matched_common_and_router', 'tied_embeddings',
                'deterministic_initialization', 'dense_manual_forward', 'sparse_legacy_parity',
                'training_inference_parity', 'finite_backward_and_parameter_update', 'rank1_actual_storage']}
