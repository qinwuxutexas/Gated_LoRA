"""
Local CPU smoke tests. No GPU, no Qwen3-VL-8B, no Colab job.

    python scripts/test_cpu.py
    python scripts/ctl.py cpu-test
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("CPU torch is not installed. In this terminal run:")
        print(
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
        raise SystemExit(2)


def _tiny_vl():
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.k_proj = nn.Linear(8, 8, bias=False)
            self.down_proj = nn.Linear(8, 8, bias=False)

    class TinyVL(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Block()
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([Block(), Block()])

    return TinyVL()


def test_gated_forward_and_grads():
    import torch
    import torch.nn as nn

    from src.gated_lora import GatedLoRALinear

    base = nn.Linear(8, 8, bias=False)
    layer = GatedLoRALinear(base, rank=4, alpha=8.0, dropout=0.0, gate_init=2.0)
    layer.eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = base(x)
        got = layer(x)
    if not torch.allclose(got, expected, atol=1e-5):
        raise AssertionError("zero-init B should leave y == Wx in eval mode")

    gates = layer.gate_values()
    if abs(float(gates.mean().detach()) - 0.880797) > 1e-3:
        raise AssertionError(f"sigmoid(2) expected ~0.881, got {float(gates.mean())}")

    layer.train()
    y = layer(x).sum()
    y.backward()
    if layer.lora_A.weight.grad is None or layer.lora_B.weight.grad is None:
        raise AssertionError("A/B should receive gradients")
    if layer.gate_logits.grad is None:
        raise AssertionError("gate_logits should receive gradients")
    if any(p.grad is not None for p in layer.base_layer.parameters()):
        raise AssertionError("base W should stay frozen")
    print("ok  gated forward / backward")


def test_inject_save_load():
    import torch

    from src.gated_lora import (
        GatedLoRALinear,
        configure_trainable_parameters,
        gated_adapter_state_dict,
        inject_gated_lora,
        load_gated_adapter,
        save_gated_adapter,
    )

    model = _tiny_vl()
    names = inject_gated_lora(model, rank=4, alpha=8.0, dropout=0.0)
    if len(names) != 6:
        raise AssertionError(f"expected 6 language targets, got {names}")
    if any("visual" in name for name in names):
        raise AssertionError(f"vision layer was injected: {names}")

    configure_trainable_parameters(model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise AssertionError("no trainable adapter params")
    if any("visual" in n and p.requires_grad for n, p in model.named_parameters()):
        raise AssertionError("vision params should stay frozen")

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GatedLoRALinear):
                module.lora_B.weight.fill_(0.1)
                module.gate_logits.fill_(-1.5)

    class DummyProcessor:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "processor.txt").write_text("dummy", encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "adapter"
        save_gated_adapter(
            model,
            DummyProcessor(),
            out,
            {"rank": 4, "alpha": 8.0},
        )
        fresh = _tiny_vl()
        inject_gated_lora(fresh, rank=4, alpha=8.0, dropout=0.0)
        load_gated_adapter(fresh, out)
        old = gated_adapter_state_dict(model)
        new = gated_adapter_state_dict(fresh)
        if set(old) != set(new):
            raise AssertionError("adapter key mismatch after reload")
        for key in old:
            if not torch.allclose(old[key], new[key]):
                raise AssertionError(f"reload mismatch: {key}")
    print("ok  inject / save / load")


def test_data_and_collator():
    import torch

    from src.data_utils import (
        ManifestDataset,
        Qwen3VLCollator,
        answer_from_record,
        question_from_record,
        resolve_image_path,
    )

    rec_ok = {
        "question": "What color?",
        "answer": "red",
        "image": "/content/drive/MyDrive/gated_lora/gqa/images/x.jpg",
    }
    rec_skip = {"question": None, "answer": None, "image": None, "metadata": {}}
    if question_from_record(rec_ok) != "What color?":
        raise AssertionError("question parse failed")
    if answer_from_record(rec_skip) is not None:
        raise AssertionError("null answer should skip")

    with tempfile.TemporaryDirectory() as tmp:
        from PIL import Image as PILImage

        data_root = Path(tmp) / "data"
        image = data_root / "gqa" / "images" / "x.jpg"
        image.parent.mkdir(parents=True)
        PILImage.new("RGB", (8, 8), (255, 0, 0)).save(image)
        resolved = resolve_image_path(rec_ok["image"], data_root)
        if Path(resolved) != image:
            raise AssertionError(f"path remap failed: {resolved}")

        manifest = Path(tmp) / "tiny.jsonl"
        rows = [
            rec_ok,
            rec_skip,
            {
                "question": "ok",
                "answer": "yes",
                "image": "/content/drive/MyDrive/gated_lora/gqa/images/x.jpg",
            },
        ]
        manifest.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )
        missing = resolve_image_path(
            "/content/drive/MyDrive/gated_lora/chartqa/images/nope.jpg",
            data_root,
        )
        expected_missing = data_root / "chartqa" / "images" / "nope.jpg"
        if Path(missing) != expected_missing:
            raise AssertionError(
                f"missing files should still remap to data/: {missing}"
            )

        rows.append({
            "question": "missing img",
            "answer": "skip",
            "image": "/content/drive/MyDrive/gated_lora/gqa/images/does_not_exist.jpg",
        })
        manifest.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )
        ds = ManifestDataset(manifest, data_root=data_root)
        if len(ds) != 3:
            raise AssertionError(f"expected 3 usable rows, got {len(ds)}")
        sample = ds[2]
        if sample["question"] not in {"What color?", "ok"}:
            raise AssertionError("missing image should skip to a readable neighbor")

    squeezed = Qwen3VLCollator._squeeze_feature(
        "image_grid_thw",
        torch.tensor([[1, 2, 3]]),
    )
    if tuple(squeezed.shape) != (1, 3):
        raise AssertionError(f"image_grid_thw should stay 2D, got {tuple(squeezed.shape)}")
    print("ok  data parse / path remap / collator squeeze")


def test_metrics():
    from src.metrics import chartqa_score, docvqa_score, gqa_score, textvqa_score

    if gqa_score("Red.", "red") != 1.0:
        raise AssertionError("gqa normalize/exact match failed")
    if textvqa_score("cat", ["cat", "cat", "cat"]) != 1.0:
        raise AssertionError("textvqa consensus of 3/3 failed")
    if abs(textvqa_score("cat", ["cat", "cat", "dog"]) - (2.0 / 3.0)) > 1e-6:
        raise AssertionError("textvqa consensus of 2/3 failed")
    if chartqa_score("21", "20") != 1.0:
        raise AssertionError("chartqa 5% relaxed match failed")
    if docvqa_score("invoice 12", "invoice 12") != 1.0:
        raise AssertionError("docvqa exact ANLS failed")
    print("ok  metrics")


def test_job_queue_isolated():
    from src.jobs import get_job, new_job, next_queued_job

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg_dir = root / "workspace"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(
            (ROOT / "workspace" / "config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        job = new_job("gated_train", root=root)
        if job["status"] != "queued":
            raise AssertionError("new job should be queued")
        loaded = get_job(job["id"], root=root)
        nxt = next_queued_job(root=root)
        if nxt is None or nxt["id"] != job["id"]:
            raise AssertionError("isolated queue did not see the test job")
        if loaded["params"]["output_dir"].endswith("gated_qlora_r8") is False:
            raise AssertionError("gated_train defaults were not applied")
    print("ok  job queue (temp dir, not Drive)")


def main() -> int:
    require_torch()
    tests = [
        test_gated_forward_and_grads,
        test_inject_save_load,
        test_data_and_collator,
        test_metrics,
        test_job_queue_isolated,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        print(f"\n{failed}/{len(tests)} failed")
        return 1
    print(f"\n{len(tests)}/{len(tests)} passed  (CPU only; 8B train still needs Colab)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
