from __future__ import annotations

from dataclasses import dataclass

from domain.localization import LocalizationService
from domain.ui_text import UICopyProfileCatalog, UITextModifier, UITextRequest


@dataclass(slots=True)
class LocalizationModifier(UITextModifier):
    localization: LocalizationService

    def modify(self, *, request: UITextRequest, text: str) -> str:
        return self.localization.t(
            request.localization_key,
            locale=request.locale,
            params=request.params,
            default=text,
        )


@dataclass(slots=True)
class CopyProfileModifier(UITextModifier):
    profiles: UICopyProfileCatalog

    def modify(self, *, request: UITextRequest, text: str) -> str:
        default_profile = self.profiles.default_profile()
        profile_candidates: list[str] = [default_profile]
        if request.profile and request.profile != default_profile:
            profile_candidates.insert(0, request.profile)

        for profile_name in profile_candidates:
            profile = self.profiles.get_profile(profile_name)
            if profile is None or not profile.enabled:
                continue
            if request.locale:
                locale_payload = profile.locale_overrides.get(request.locale)
                if locale_payload is not None:
                    locale_text = locale_payload.get(request.text_id)
                    if locale_text is not None:
                        return locale_text
            override = profile.overrides.get(request.text_id)
            if override is not None:
                return override
        return text


def create_localization_modifier(
    *,
    localization: LocalizationService,
    **_: object,
) -> UITextModifier:
    return LocalizationModifier(localization=localization)


def create_copy_profile_modifier(
    *,
    copy_profiles: UICopyProfileCatalog,
    **_: object,
) -> UITextModifier:
    return CopyProfileModifier(profiles=copy_profiles)
