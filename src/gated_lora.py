"""
Gated LoRA: y = W x + (alpha/r) * B (g ⊙ (A x))

Standard LoRA is B(Ax). The extra parameter is a per-rank gate g = sigmoid(gate_logits).
Vision tower stays frozen; adapters attach to language-model projections only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

TARGET_LEAF_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class GatedLoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16.0, dropout=0.05, gate_init=2.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate_logits = nn.Parameter(
            torch.full((self.rank,), float(gate_init), dtype=torch.float32)
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def gate_values(self):
        return torch.sigmoid(self.gate_logits)

    def forward(self, x):
        base_output = self.base_layer(x)
        adapter_dtype = self.lora_A.weight.dtype
        z = self.lora_A(self.dropout(x).to(adapter_dtype))
        z = z * self.gate_values().to(dtype=z.dtype, device=z.device)
        delta = self.lora_B(z) * self.scaling
        return base_output + delta.to(base_output.dtype)


def get_parent_module(model, module_name):
    parts = module_name.split(".")
    child_name = parts[-1]
    parent_name = ".".join(parts[:-1])
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, child_name


def is_language_target(module_name, module):
    leaf = module_name.split(".")[-1]
    if leaf not in TARGET_LEAF_NAMES:
        return False
    name = module_name.lower()
    if "visual" in name:
        return False
    if "language_model" not in module_name:
        return False
    return hasattr(module, "in_features") and hasattr(module, "out_features")


def inject_gated_lora(model, rank=8, alpha=16.0, dropout=0.05, gate_init=2.0):
    targets = [
        (name, module)
        for name, module in list(model.named_modules())
        if is_language_target(name, module)
    ]
    if not targets:
        raise RuntimeError(
            "Could not locate Qwen3-VL language-model projection layers."
        )

    print(f"Found {len(targets)} language linear layers for gated LoRA.")
    for module_name, base_module in targets:
        parent, child_name = get_parent_module(model, module_name)
        setattr(
            parent,
            child_name,
            GatedLoRALinear(
                base_layer=base_module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                gate_init=gate_init,
            ),
        )
    return [name for name, _ in targets]


def mark_quantized_model_has_adapters(model):
    """
    HuggingFace Trainer blocks training a 4-bit model unless PEFT adapters are attached.
    Gated LoRA is a custom adapter, so set the same flag transformers uses for PEFT.
    """
    model._hf_peft_config_loaded = True
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Gated LoRA injected but no parameter requires grad.")


def configure_trainable_parameters(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            ".lora_A." in name
            or ".lora_B." in name
            or name.endswith("gate_logits")
        )


def adapter_parameter_summary(model):
    total = trainable = adapter = gate_params = 0
    for name, parameter in model.named_parameters():
        n = parameter.numel()
        total += n
        if parameter.requires_grad:
            trainable += n
            if "gate_logits" in name:
                gate_params += n
            else:
                adapter += n
    return {
        "total": total,
        "trainable": trainable,
        "adapter": adapter,
        "gate_params": gate_params,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }


def print_trainable_parameters(model):
    s = adapter_parameter_summary(model)
    print("PARAMETER SUMMARY")
    print(f"Total parameters:     {s['total']:,}")
    print(f"Trainable parameters: {s['trainable']:,}")
    print(f"A/B parameters:       {s['adapter']:,}")
    print(f"Gate parameters:      {s['gate_params']:,}")
    print(f"Trainable percentage: {s['trainable_pct']:.6f}%")


def get_gate_statistics(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, GatedLoRALinear):
            gates = module.gate_values().detach().float().cpu()
            rows.append({
                "module": name,
                "rank": module.rank,
                "mean_gate": float(gates.mean()),
                "min_gate": float(gates.min()),
                "max_gate": float(gates.max()),
                "active_frac_0.5": float((gates >= 0.5).float().mean()),
                "gates": gates.tolist(),
            })
    return rows


def print_gate_summary(model):
    stats = get_gate_statistics(model)
    if not stats:
        return
    all_gates = torch.tensor(
        [g for row in stats for g in row["gates"]],
        dtype=torch.float32,
    )
    print("GATE SUMMARY")
    print(f"Gated layers: {len(stats)}")
    print(f"Mean gate: {all_gates.mean().item():.6f}")
    print(f"Min gate:  {all_gates.min().item():.6f}")
    print(f"Max gate:  {all_gates.max().item():.6f}")
    for threshold in (0.1, 0.25, 0.5, 0.75):
        frac = (all_gates >= threshold).float().mean().item()
        print(f"Fraction gate >= {threshold:.2f}: {frac:.4f}")


def gated_adapter_state_dict(model):
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name or name.endswith("gate_logits")
    }


def save_gated_adapter(model, processor, output_dir, config_dict):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gated_adapter_state_dict(model), output_dir / "gated_adapter.pt")
    (output_dir / "gated_config.json").write_text(
        json.dumps(config_dict, indent=2),
        encoding="utf-8",
    )
    (output_dir / "gate_statistics.json").write_text(
        json.dumps(get_gate_statistics(model), indent=2),
        encoding="utf-8",
    )
    processor.save_pretrained(output_dir)
    print("Saved gated adapter to:", output_dir / "gated_adapter.pt")


def load_gated_adapter(model, adapter_path):
    adapter_path = Path(adapter_path)
    if adapter_path.is_dir():
        adapter_path = adapter_path / "gated_adapter.pt"
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected gated adapter keys: {unexpected[:20]}")
    print("Loaded gated adapter:", adapter_path)


def move_adapters_to_dtype(model, dtype=torch.bfloat16):
    for module in model.modules():
        if isinstance(module, GatedLoRALinear):
            module.lora_A = module.lora_A.to(dtype=dtype)
            module.lora_B = module.lora_B.to(dtype=dtype)
