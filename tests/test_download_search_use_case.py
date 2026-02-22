from __future__ import annotations

from application.download_search_use_case import DownloadSearchUseCase


def _use_case() -> DownloadSearchUseCase:
    return DownloadSearchUseCase(
        base_code_to_api={
            "all": [],
            "sdxl": ["SDXL 1.0"],
            "flux": ["Flux Dev"],
        },
        sort_code_to_api={
            "downloads": "Most Downloaded",
            "rating": "Highest Rated",
        },
        period_code_to_api={
            "all": "AllTime",
            "month": "Month",
        },
        supports_base_filter=lambda *, model_type, source: model_type == "checkpoint"
        and source in {"all", "civitai"},
        supports_nsfw_filter=lambda source: source in {"all", "civitai"},
    )


def test_build_search_criteria_maps_state_to_search_payload() -> None:
    use_case = _use_case()
    criteria = use_case.build_search_criteria(
        {
            "dl_query": "  anime portrait  ",
            "dl_type": "checkpoint",
            "dl_source": "all",
            "dl_sort": "rating",
            "dl_period": "month",
            "dl_base": "sdxl",
            "dl_nsfw": True,
            "dl_author": "@Alice, bob,alice",
            "dl_page_size": "5",
        },
        requested_limit=3,
        default_page_size=8,
    )

    assert criteria.query == "anime portrait"
    assert criteria.model_type == "checkpoint"
    assert criteria.source == "all"
    assert criteria.limit == 5
    assert criteria.sort == "Highest Rated"
    assert criteria.base_models == ("SDXL 1.0",)
    assert criteria.include_nsfw is True
    assert criteria.period == "Month"
    assert criteria.civitai_authors == ("Alice", "bob")


def test_build_search_criteria_disables_unsupported_filters() -> None:
    use_case = _use_case()
    criteria = use_case.build_search_criteria(
        {
            "dl_type": "lora",
            "dl_source": "huggingface",
            "dl_base": "flux",
            "dl_nsfw": True,
            "dl_page_size": 0,
        },
        requested_limit=7,
        default_page_size=8,
    )

    assert criteria.limit == 8
    assert criteria.base_models == ()
    assert criteria.include_nsfw is False
    assert criteria.civitai_authors == ()


def test_infer_base_code_for_checkpoint_uses_metadata_or_fallback() -> None:
    use_case = _use_case()

    assert (
        use_case.infer_base_code_for_checkpoint(
            "sdxl_model.safetensors",
            get_model_metadata=lambda filename, *, model_type=None: {"base_model": "SDXL 1.0"},
            infer_base_model=lambda _: "",
        )
        == "sdxl"
    )

    assert (
        use_case.infer_base_code_for_checkpoint(
            "flux_model.safetensors",
            get_model_metadata=lambda filename, *, model_type=None: None,
            infer_base_model=lambda _: "Flux Dev",
        )
        == "flux"
    )

    assert (
        use_case.infer_base_code_for_checkpoint(
            "",
            get_model_metadata=lambda filename, *, model_type=None: None,
            infer_base_model=lambda _: "",
            default="all",
        )
        == "all"
    )
