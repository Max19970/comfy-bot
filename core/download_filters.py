from __future__ import annotations


def base_code_from_base_model(base_model: str) -> str:
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
