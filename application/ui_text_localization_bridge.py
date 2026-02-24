from __future__ import annotations

from collections.abc import Mapping

from domain.localization import LocalizationService
from domain.ui_text import UITextService, normalize_text_id


class UITextLocalizationBridge(LocalizationService):
    def __init__(self, *, localization: LocalizationService, ui_text: UITextService) -> None:
        self._localization = localization
        self._ui_text = ui_text

    def t(
        self,
        key: str,
        *,
        locale: str | None = None,
        params: Mapping[str, object] | None = None,
        default: str | None = None,
    ) -> str:
        normalized_key = normalize_text_id(key, default="")
        if not normalized_key:
            return self._localization.t(
                key,
                locale=locale,
                params=params,
                default=default,
            )
        return self._ui_text.text(
            normalized_key,
            locale=locale,
            params=params,
            default=default,
        )

    def default_locale(self) -> str:
        return self._localization.default_locale()

    def available_locales(self) -> tuple[str, ...]:
        return self._localization.available_locales()
