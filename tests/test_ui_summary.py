from __future__ import annotations

from core.models import GenerationParams
from core.ui import (
    MAX_REFERENCE_IMAGES,
    loras_text,
    params_summary_for_mode,
    params_summary_full,
    params_summary_simple,
)


def test_loras_text_escapes_html_and_formats_strength() -> None:
    rendered = loras_text([("name<lora>.safetensors", 0.75)])

    assert "name&lt;lora&gt;.safetensors" in rendered
    assert "<code>0.75</code>" in rendered


def test_params_summary_simple_shows_key_fields() -> None:
    params = GenerationParams(
        checkpoint="pony-model.safetensors",
        positive="a detailed portrait",
        negative="blurry",
        width=768,
        height=1024,
        loras=[("style.safetensors", 0.8)],
        enable_hires_fix=True,
        enable_pag=True,
    )

    summary = params_summary_simple(params)

    assert "🎨 <b>pony-model.safetensors</b> | 768×1024" in summary
    assert "🟢 <b>Positive:</b>" in summary
    assert "🔴 <b>Negative:</b>" in summary
    assert "📎 <b>LoRA:</b>" in summary
    assert "✨ <b>Улучшения:</b>" in summary


def test_params_summary_full_shows_reference_counter_and_sampling_block() -> None:
    params = GenerationParams(
        checkpoint="sdxl.safetensors",
        positive="cat",
        negative="lowres",
        reference_images=[{"id": "r1", "file_id": "f1"}, {"id": "r2", "file_id": "f2"}],
        sampler="euler",
        scheduler="normal",
        cfg=6.5,
        denoise=0.8,
        seed=123,
    )

    summary = params_summary_full(params)

    assert f"🖼 <b>Ref:</b> 2/{MAX_REFERENCE_IMAGES}" in summary
    assert "⚙️ <b>Sampler:</b> <code>euler</code>" in summary
    assert "<b>Sched:</b> <code>normal</code>" in summary
    assert "<b>Seed:</b> <code>123</code>" in summary


def test_params_summary_for_mode_switches_between_simple_and_full() -> None:
    params = GenerationParams(checkpoint="sdxl.safetensors", positive="cat", negative="bad")

    simple = params_summary_for_mode(params, pro_mode=False)
    full = params_summary_for_mode(params, pro_mode=True)

    assert simple == params_summary_simple(params)
    assert full == params_summary_full(params)
    assert simple != full
