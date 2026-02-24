from __future__ import annotations

from dataclasses import dataclass

import pytest

from domain.ui_text import UICopyProfile
from infrastructure.ui_text_modifiers import CopyProfileModifier, LocalizationModifier
from infrastructure.ui_text_pipeline import UITextPipelineError, build_ui_text_modifiers


@dataclass(slots=True)
class _StubLocalization:
    def t(
        self,
        key: str,
        *,
        locale: str | None = None,
        params: dict[str, object] | None = None,
        default: str | None = None,
    ) -> str:
        return default or key


@dataclass(slots=True)
class _StubProfiles:
    def list_profiles(self) -> list[str]:
        return ["base"]

    def default_profile(self) -> str:
        return "base"

    def get_profile(self, profile: str) -> UICopyProfile | None:
        if profile != "base":
            return None
        return UICopyProfile(
            profile="base",
            enabled=True,
            is_default=True,
            overrides={},
            locale_overrides={},
        )


def test_build_ui_text_modifiers_builds_ordered_pipeline() -> None:
    modifiers = build_ui_text_modifiers(
        (
            "infrastructure.ui_text_modifiers:create_localization_modifier,"
            "infrastructure.ui_text_modifiers:create_copy_profile_modifier"
        ),
        dependencies={
            "localization": _StubLocalization(),
            "copy_profiles": _StubProfiles(),
        },
    )

    assert len(modifiers) == 2
    assert isinstance(modifiers[0], LocalizationModifier)
    assert isinstance(modifiers[1], CopyProfileModifier)


def test_build_ui_text_modifiers_fails_on_missing_dependency() -> None:
    with pytest.raises(UITextPipelineError):
        build_ui_text_modifiers(
            "infrastructure.ui_text_modifiers:create_copy_profile_modifier",
            dependencies={"localization": _StubLocalization()},
        )
