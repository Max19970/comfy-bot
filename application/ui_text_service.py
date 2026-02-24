from __future__ import annotations

from collections.abc import Mapping, Sequence

from domain.ui_text import (
    UICopyProfileCatalog,
    UITextModifier,
    UITextRegistry,
    UITextRequest,
    UITextService,
    normalize_locale_or_empty,
    normalize_profile_name,
    normalize_text_id,
)


class _SafeFormatDict(dict[str, object]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class DefaultUITextService(UITextService):
    def __init__(
        self,
        *,
        registry: UITextRegistry,
        profiles: UICopyProfileCatalog,
        modifiers: Sequence[UITextModifier] | None = None,
    ) -> None:
        self._registry = registry
        self._profiles = profiles
        self._modifiers = tuple(modifiers or ())

    def default_profile(self) -> str:
        return self._profiles.default_profile()

    def available_profiles(self) -> tuple[str, ...]:
        return tuple(self._profiles.list_profiles())

    def text(
        self,
        text_id: str,
        *,
        locale: str | None = None,
        profile: str | None = None,
        params: Mapping[str, object] | None = None,
        default: str | None = None,
    ) -> str:
        normalized_text_id = normalize_text_id(text_id, default="")
        if not normalized_text_id:
            return _safe_format(default or "", params)

        resolved_locale = normalize_locale_or_empty(locale)
        resolved_profile = normalize_profile_name(profile or "", default=self.default_profile())

        definition = self._registry.get_definition(normalized_text_id)
        default_text = default
        localization_key = normalized_text_id

        if definition is not None:
            if default_text is None:
                default_text = definition.default_text
            localization_key = definition.localization_key or definition.text_id

        if default_text is None:
            default_text = normalized_text_id

        request = UITextRequest(
            text_id=normalized_text_id,
            localization_key=localization_key,
            default_text=default_text,
            locale=resolved_locale or None,
            profile=resolved_profile,
            params=params,
        )

        text = default_text
        for modifier in self._modifiers:
            text = modifier.modify(request=request, text=text)

        return _safe_format(text, params)


def _safe_format(template: str, params: Mapping[str, object] | None) -> str:
    if not params:
        return template
    normalized_params = _SafeFormatDict({str(key): value for key, value in params.items()})
    try:
        return template.format_map(normalized_params)
    except (ValueError, TypeError):
        return template
