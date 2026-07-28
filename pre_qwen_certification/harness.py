"""Capture/replay and adversarial fault-injection harness."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from .metrics import TensorMetrics, tensor_metrics
from .modal import Routing


FaultName = Literal[
    "none",
    "shift-input",
    "permute-routes",
    "reverse-route-weights",
    "wrong-input-scale",
    "double-residual",
]


@dataclass
class Capture:
    inputs: torch.Tensor
    outputs: torch.Tensor
    router_logits: torch.Tensor
    top_ids: torch.Tensor
    route_weights: torch.Tensor
    residual: torch.Tensor | None
    document_ids: list[str]
    sequence_ids: list[str]
    token_positions: torch.Tensor
    layer_id: int

    def validate(self) -> None:
        tokens = self.inputs.shape[0]
        if self.inputs.ndim != 2 or self.outputs.ndim != 2:
            raise ValueError("inputs and outputs must be [tokens, hidden]")
        if self.outputs.shape != self.inputs.shape:
            raise ValueError("capture output must preserve hidden shape")
        if self.top_ids.shape != self.route_weights.shape:
            raise ValueError("top_ids and route_weights must have equal shapes")
        if self.top_ids.shape[0] != tokens:
            raise ValueError("routing token count does not match inputs")
        if self.router_logits.shape[0] != tokens:
            raise ValueError("router logits token count does not match inputs")
        if self.residual is not None and self.residual.shape != self.inputs.shape:
            raise ValueError("residual shape mismatch")
        if len(self.document_ids) != tokens or len(self.sequence_ids) != tokens:
            raise ValueError("metadata row count mismatch")
        if tuple(self.token_positions.shape) != (tokens,):
            raise ValueError("token_positions must have shape [tokens]")


def capture_layer(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    document_ids: list[str],
    sequence_ids: list[str],
    token_positions: torch.Tensor,
    layer_id: int,
    residual: torch.Tensor | None = None,
) -> Capture:
    model.eval()
    with torch.no_grad():
        output, routing = model(inputs)
    if not isinstance(routing, Routing):
        raise TypeError("certification models must return a Routing object")
    capture = Capture(
        inputs=inputs.detach().cpu(),
        outputs=output.detach().cpu(),
        router_logits=routing.logits.detach().cpu(),
        top_ids=routing.top_ids.detach().cpu(),
        route_weights=routing.weights.detach().cpu(),
        residual=None if residual is None else residual.detach().cpu(),
        document_ids=list(document_ids),
        sequence_ids=list(sequence_ids),
        token_positions=token_positions.detach().cpu(),
        layer_id=int(layer_id),
    )
    capture.validate()
    return capture


def save_capture(path: Path, capture: Capture) -> dict[str, object]:
    capture.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_payload = {
        "inputs": capture.inputs,
        "outputs": capture.outputs,
        "router_logits": capture.router_logits,
        "top_ids": capture.top_ids,
        "route_weights": capture.route_weights,
        "residual": capture.residual,
        "token_positions": capture.token_positions,
    }
    torch.save(tensor_payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "sha256": digest,
        "layer_id": capture.layer_id,
        "document_ids": capture.document_ids,
        "sequence_ids": capture.sequence_ids,
        "tokens": capture.inputs.shape[0],
        "hidden_size": capture.inputs.shape[1],
        "top_k": capture.top_ids.shape[1],
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_capture(path: Path) -> Capture:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("capture digest mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    capture = Capture(
        inputs=payload["inputs"],
        outputs=payload["outputs"],
        router_logits=payload["router_logits"],
        top_ids=payload["top_ids"],
        route_weights=payload["route_weights"],
        residual=payload.get("residual"),
        document_ids=list(metadata["document_ids"]),
        sequence_ids=list(metadata["sequence_ids"]),
        token_positions=payload["token_positions"],
        layer_id=int(metadata["layer_id"]),
    )
    capture.validate()
    return capture


def _faulted_inputs(capture: Capture, fault: FaultName) -> torch.Tensor:
    inputs = capture.inputs.clone()
    if fault == "shift-input":
        return torch.roll(inputs, shifts=1, dims=0)
    if fault == "wrong-input-scale":
        return inputs * 1.125
    return inputs


def _faulted_routing(capture: Capture, fault: FaultName) -> tuple[torch.Tensor, torch.Tensor]:
    top_ids = capture.top_ids.clone()
    weights = capture.route_weights.clone()
    if fault == "permute-routes":
        permutation = torch.arange(top_ids.max().item() + 1)
        permutation = torch.roll(permutation, shifts=1)
        top_ids = permutation[top_ids]
    elif fault == "reverse-route-weights":
        weights = torch.flip(weights, dims=(-1,))
    return top_ids, weights


def replay_layer(
    model: nn.Module,
    capture: Capture,
    *,
    fault: FaultName = "none",
) -> tuple[torch.Tensor, TensorMetrics]:
    capture.validate()
    model.eval()
    inputs = _faulted_inputs(capture, fault)
    top_ids, weights = _faulted_routing(capture, fault)
    with torch.no_grad():
        output, _ = model(
            inputs,
            forced_top_ids=top_ids,
            forced_weights=weights,
        )
    if capture.residual is not None:
        output = output + capture.residual
        target = capture.outputs + capture.residual
        if fault == "double-residual":
            output = output + capture.residual
    else:
        target = capture.outputs
        if fault == "double-residual":
            # A deterministic nonzero surrogate catches accidental residual addition
            # even when the capture intentionally omitted one.
            output = output + capture.inputs
    metrics = tensor_metrics(output, target)
    return output, metrics


def run_fault_matrix(
    model: nn.Module,
    capture: Capture,
) -> dict[str, dict[str, float]]:
    faults: tuple[FaultName, ...] = (
        "none",
        "shift-input",
        "permute-routes",
        "reverse-route-weights",
        "wrong-input-scale",
        "double-residual",
    )
    return {
        fault: replay_layer(model, capture, fault=fault)[1].as_dict()
        for fault in faults
    }
