from __future__ import annotations

from collections.abc import Mapping

from domain.localization import LocalizationService, TranslationCatalog, normalize_locale_code


class _SafeFormatDict(dict[str, object]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class DefaultLocalizationService(LocalizationService):
    def __init__(self, catalog: TranslationCatalog) -> None:
        self._catalog = catalog

    def default_locale(self) -> str:
        return self._catalog.default_locale()

    def available_locales(self) -> tuple[str, ...]:
        return tuple(self._catalog.list_locales())

    def t(
        self,
        key: str,
        *,
        locale: str | None = None,
        params: Mapping[str, object] | None = None,
        default: str | None = None,
    ) -> str:
        normalized_key = str(key or "").strip()
        if not normalized_key:
            return default or ""

        resolved_locale = normalize_locale_code(
            str(locale or ""),
            default=self.default_locale(),
        )
        for locale_code in self._locale_chain(resolved_locale):
            bundle = self._catalog.get_bundle(locale_code)
            if bundle is None:
                continue
            translated = bundle.messages.get(normalized_key)
            if translated is None:
                continue
            if params:
                return _safe_format(translated, params)
            return translated

        if default is not None:
            return default
        return normalized_key

    def _locale_chain(self, start_locale: str) -> tuple[str, ...]:
        default_locale = self.default_locale()
        visited: set[str] = set()
        chain: list[str] = []

        current = normalize_locale_code(start_locale, default=default_locale)
        while current and current not in visited:
            visited.add(current)
            chain.append(current)

            bundle = self._catalog.get_bundle(current)
            if bundle is None:
                break

            fallback_locale = normalize_locale_code(
                str(bundle.metadata.fallback_locale or ""),
                default="",
            )
            if not fallback_locale or fallback_locale in visited:
                break
            current = fallback_locale

        if default_locale and default_locale not in visited:
            chain.append(default_locale)

        return tuple(chain)


def _safe_format(template: str, params: Mapping[str, object]) -> str:
    normalized_params = _SafeFormatDict({str(key): value for key, value in params.items()})
    try:
        return template.format_map(normalized_params)
    except (ValueError, TypeError):
        return template
