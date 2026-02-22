from __future__ import annotations

import asyncio
import json
from pathlib import Path

from infrastructure.model_metadata_index import ModelMetadataIndexRepository


def _index_path(root: Path) -> Path:
    return root / ".comfybot_model_index.json"


def test_model_metadata_index_loads_existing_payload(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "models": {
            "anime.safetensors": {
                "filename": "anime.safetensors",
                "model_type": "lora",
                "base_model": "SDXL",
            }
        },
    }
    _index_path(tmp_path).write_text(json.dumps(payload), encoding="utf-8")

    repo = ModelMetadataIndexRepository(str(tmp_path))
    metadata = repo.get("anime.safetensors", model_type="lora")

    assert metadata is not None
    assert metadata["base_model"] == "SDXL"


def test_model_metadata_index_upsert_supports_aliases(tmp_path: Path) -> None:
    repo = ModelMetadataIndexRepository(str(tmp_path))

    asyncio.run(
        repo.upsert(
            "Anime.Safetensors",
            {
                "filename": "Anime.Safetensors",
                "model_type": "lora",
                "trained_words": ["anime_style"],
            },
            aliases=["anime_original.safetensors"],
        )
    )

    by_target = repo.get("anime.safetensors", model_type="lora")
    by_alias = repo.get("anime_original.safetensors", model_type="lora")
    assert by_target is not None
    assert by_alias is not None
    assert by_target["trained_words"] == ["anime_style"]
    assert by_alias["trained_words"] == ["anime_style"]

    persisted = json.loads(_index_path(tmp_path).read_text(encoding="utf-8"))
    assert isinstance(persisted.get("models"), dict)


def test_model_metadata_index_delete_removes_key(tmp_path: Path) -> None:
    repo = ModelMetadataIndexRepository(str(tmp_path))
    asyncio.run(
        repo.upsert(
            "anime.safetensors",
            {
                "filename": "anime.safetensors",
                "model_type": "lora",
            },
        )
    )

    assert repo.get("anime.safetensors") is not None
    asyncio.run(repo.delete("anime.safetensors"))
    assert repo.get("anime.safetensors") is None
