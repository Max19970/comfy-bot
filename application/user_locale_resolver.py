from __future__ import annotations

from dataclasses import dataclass

from domain.localization import LocalizationService, UserLocaleResolver, normalize_locale_code


@dataclass(slots=True)
class DefaultUserLocaleResolver(UserLocaleResolver):
    localization: LocalizationService

    def resolve(
        self,
        *,
        user_locale: str | None,
        telegram_locale: str | None,
    ) -> str:
        default_locale = self.localization.default_locale()
        available = set(self.localization.available_locales())

        preferred = normalize_locale_code(str(user_locale or ""), default="")
        if preferred and preferred in available:
            return preferred

        telegram_candidates = self._telegram_locale_candidates(telegram_locale)
        for candidate in telegram_candidates:
            if candidate in available:
                return candidate

        return default_locale

    def _telegram_locale_candidates(self, telegram_locale: str | None) -> tuple[str, ...]:
        normalized = normalize_locale_code(str(telegram_locale or ""), default="")
        if not normalized:
            return ()
        parts = normalized.split("-", maxsplit=1)
        if len(parts) < 2:
            return (normalized,)
        base = parts[0]
        if base == normalized:
            return (normalized,)
        return (normalized, base)
