from __future__ import annotations


class BaseModelPolicy:
    """Heuristics for model-family detection and compatibility checks."""

    def infer_base_model(self, text: str) -> str:
        value = str(text or "").strip().lower()
        if not value:
            return ""
        if "pony" in value:
            return "Pony"
        if "flux" in value:
            return "Flux"
        if "illustrious" in value:
            return "Illustrious"
        if "sdxl" in value or "stable diffusion xl" in value:
            return "SDXL"
        if "sd 2.1" in value or "sd2.1" in value or "stable diffusion 2.1" in value:
            return "SD 2.1"
        if "sd 2" in value or "sd2" in value:
            return "SD 2.x"
        if (
            "sd 1.5" in value
            or "sd1.5" in value
            or "stable diffusion 1.5" in value
            or "_15" in value
        ):
            return "SD 1.5"
        if "sd 1" in value or "stable diffusion 1" in value:
            return "SD 1.x"
        return ""

    def base_models_compatible(self, checkpoint_base: str, lora_base: str) -> bool:
        ck = self._base_family(checkpoint_base)
        lr = self._base_family(lora_base)
        if not ck or not lr:
            return True
        if ck == lr:
            return True
        if {ck, lr} <= {"sdxl", "illustrious"}:
            return True
        return False

    def download_base_code(self, base_model: str) -> str:
        value = str(base_model or "").strip().lower()
        if not value:
            return "all"
        if "illustrious" in value:
            return "illustrious"
        if "noob" in value:
            return "noobai"
        if "pony" in value:
            return "pony"
        if "flux" in value:
            return "flux"
        if "3.5" in value:
            return "sd35"
        if "sd3" in value or "stable diffusion 3" in value:
            return "sd3"
        if "2.1" in value or "2.0" in value or "sd2" in value:
            return "sd2"
        if "0.9" in value:
            return "sdxl09"
        if "sdxl" in value or "stable diffusion xl" in value:
            return "sdxl"
        if "1.5" in value or "sd15" in value:
            return "sd15"
        return "all"

    def _base_family(self, base_model: str) -> str:
        value = str(base_model or "").strip().lower()
        if not value:
            return ""
        if "pony" in value:
            return "pony"
        if "flux" in value:
            return "flux"
        if "illustrious" in value:
            return "illustrious"
        if "sdxl" in value or "stable diffusion xl" in value:
            return "sdxl"
        if "sd 2" in value or "stable diffusion 2" in value:
            return "sd2"
        if "sd 1.5" in value or "sd1.5" in value or "stable diffusion 1.5" in value:
            return "sd15"
        if "sd 1" in value or "stable diffusion 1" in value:
            return "sd1"
        return value
