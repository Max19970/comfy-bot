"""
Standalone test script for TIPO Smart Prompt inference.
Run:  python test_smart_prompt.py

Tests the full TIPO pipeline without starting the Telegram bot.
"""

from __future__ import annotations

import sys
import time

# ---------------------------------------------------------------------------
# Minimal Config stub so SmartPromptService can be instantiated standalone
# ---------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass
class _TestConfig:
    smart_prompt_provider: str = "tipo"
    smart_prompt_model: str = "KBlueLeaf/TIPO-500M-ft"
    smart_prompt_timeout: int = 300
    smart_prompt_temperature: float = 0.35
    smart_prompt_top_p: float = 0.95
    smart_prompt_min_p: float = 0.05
    smart_prompt_top_k: int = 80
    smart_prompt_device: str = "cuda"
    smart_prompt_seed: int = 42
    smart_prompt_tag_length: str = "long"
    smart_prompt_nl_length: str = "short"
    smart_prompt_format: str = "both_tag_first"
    smart_prompt_ban_tags: str = "text, watermark, signature"
    smart_prompt_negative_base: str = (
        "worst quality, low quality, bad quality, blurry, jpeg artifacts, "
        "bad anatomy, deformed, ugly, extra limbs, missing limbs, disfigured, "
        "mutilated, out of frame, duplicate, watermark, signature, text"
    )


# Patch config module so smart_prompt can import Config
sys.modules.setdefault("config", type(sys)("config"))
sys.modules["config"].Config = _TestConfig  # type: ignore[attr-defined]

from smart_prompt import SmartPromptService, SmartPromptError  # noqa: E402

# ---------------------------------------------------------------------------
CONTROL_MARKERS = ("<|", "|>", "<0x", "target:", "target :")

TEST_CASES = [
    {
        "description": "a girl with blue hair standing in a flower field at sunset",
        "checkpoint": "waiIllustriousSDXL_v160.safetensors",
        "expect_tags": ["girl", "hair", "flower"],
    },
    {
        "description": "кот в шляпе сидит на подоконнике",
        "checkpoint": "animagineXL_v31.safetensors",
        "expect_tags": ["cat"],
    },
    {
        "description": "a cyberpunk cityscape at night with neon lights and rain",
        "checkpoint": "ponyDiffusionV6XL_v6.safetensors",
        "expect_tags": ["city", "neon", "night"],
    },
]


def run_test(svc: SmartPromptService, case: dict) -> bool:
    desc = case["description"]
    ckpt = case["checkpoint"]
    expect = case.get("expect_tags", [])

    print(f"\n{'=' * 70}")
    print(f"  Input:      {desc}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Quality:    {svc.detect_quality_preset(ckpt)}")
    print(f"{'=' * 70}")

    t0 = time.perf_counter()
    try:
        result = svc._generate_prompts_sync(description=desc, checkpoint=ckpt)
    except SmartPromptError as e:
        print(f"  ❌ ERROR: {e}")
        return False
    elapsed = time.perf_counter() - t0

    print(f"  ⏱  Time: {elapsed:.2f}s")
    print(f"  ✅ Positive ({len(result.positive)} chars):")
    print(f"     {result.positive[:500]}")
    if len(result.positive) > 500:
        print(f"     ... [{len(result.positive) - 500} more chars]")
    print(f"  🚫 Negative ({len(result.negative)} chars):")
    print(f"     {result.negative[:300]}")

    # Checks
    ok = True
    lowered = result.positive.lower()

    # Check no control markers leaked
    for marker in CONTROL_MARKERS:
        if marker in lowered:
            print(f"  ❌ FAIL: control marker '{marker}' found in positive prompt!")
            ok = False

    # Check non-empty
    if not result.positive.strip():
        print("  ❌ FAIL: positive prompt is empty!")
        ok = False

    # Check expected tags (loose substring match)
    for tag in expect:
        if tag.lower() not in lowered:
            print(f"  ⚠️  WARN: expected '{tag}' not found in positive prompt")

    if ok:
        print("  ✅ PASS")
    return ok


def main():
    print("=" * 70)
    print("  TIPO Smart Prompt Test")
    print("=" * 70)

    cfg = _TestConfig()
    svc = SmartPromptService(cfg)

    # Check dependencies
    if not svc.is_enabled():
        hint = svc.configuration_hint()
        print(f"❌ Smart Prompt not enabled: {hint}")
        sys.exit(1)

    # Load model
    print("\n⏳ Loading TIPO model...")
    t0 = time.perf_counter()
    try:
        svc._ensure_backend_loaded_sync()
    except SmartPromptError as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    print(f"✅ Model loaded in {time.perf_counter() - t0:.1f}s")

    # Run tests
    passed = 0
    failed = 0
    for case in TEST_CASES:
        if run_test(svc, case):
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
