from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class IpAdapterNode:
    node_id = "ipadapter"
    phase = 20
    order = 30

    def apply(self, state: ComfyWorkflowState) -> None:
        if state.context.reference_mode != "ipadapter" or not state.context.reference_image_name:
            return
        if state.model_out is None:
            raise RuntimeError("IP-Adapter node requires model output")

        apply_class = state.context.info.ipadapter_apply_node
        if not apply_class:
            raise RuntimeError("IP-Adapter apply node not available")

        clip_loader_inputs = state.context.required_input_defaults("CLIPVisionLoader")
        clip_name_field = state.context.select_field_name(
            clip_loader_inputs,
            ("clip_name", "model_name"),
        )
        if not clip_name_field or not state.context.info.clip_vision_models:
            raise RuntimeError("CLIP Vision loader is not configured")
        clip_loader_inputs[clip_name_field] = state.context.info.clip_vision_models[0]
        unresolved_clip_loader = [
            name for name, value in clip_loader_inputs.items() if value is None
        ]
        if unresolved_clip_loader:
            raise RuntimeError(
                "CLIPVisionLoader has unresolved required fields: "
                + ", ".join(unresolved_clip_loader)
            )

        ip_model_loader_inputs = state.context.required_input_defaults("IPAdapterModelLoader")
        ip_model_field = state.context.select_field_name(
            ip_model_loader_inputs,
            ("ipadapter_file", "model_name", "ipadapter_name"),
        )
        if not ip_model_field or not state.context.info.ipadapter_models:
            raise RuntimeError("IP-Adapter model loader is not configured")
        ip_model_loader_inputs[ip_model_field] = state.context.info.ipadapter_models[0]
        unresolved_ip_loader = [
            name for name, value in ip_model_loader_inputs.items() if value is None
        ]
        if unresolved_ip_loader:
            raise RuntimeError(
                "IPAdapterModelLoader has unresolved required fields: "
                + ", ".join(unresolved_ip_loader)
            )

        clip_loader_id = state.add_node("CLIPVisionLoader", clip_loader_inputs)
        ip_model_loader_id = state.add_node("IPAdapterModelLoader", ip_model_loader_inputs)
        ref_image_id = state.add_node("LoadImage", {"image": state.context.reference_image_name})

        apply_inputs = state.context.required_input_defaults(apply_class)
        model_field = state.context.select_field_name(apply_inputs, ("model",))
        image_field = state.context.select_field_name(apply_inputs, ("image", "images"))
        clip_field = state.context.select_field_name(
            apply_inputs,
            ("clip_vision", "clipvision"),
        )
        ipadapter_field = state.context.select_field_name(
            apply_inputs,
            ("ipadapter", "ipadapter_model"),
        )
        if not model_field or not image_field or not clip_field or not ipadapter_field:
            raise RuntimeError("IP-Adapter apply node has unsupported input schema")

        apply_inputs[model_field] = state.model_out
        apply_inputs[image_field] = [ref_image_id, 0]
        apply_inputs[clip_field] = [clip_loader_id, 0]
        apply_inputs[ipadapter_field] = [ip_model_loader_id, 0]

        strength_field = state.context.select_field_name(
            apply_inputs,
            ("weight", "strength", "scale"),
        )
        if strength_field:
            apply_inputs[strength_field] = state.image.reference_strength

        for field_name, field_value in (
            ("start_at", 0.0),
            ("end_at", 1.0),
            ("start_at_percent", 0.0),
            ("end_at_percent", 1.0),
        ):
            if field_name in apply_inputs:
                apply_inputs[field_name] = field_value

        unresolved_apply = [name for name, value in apply_inputs.items() if value is None]
        if unresolved_apply:
            raise RuntimeError(
                f"{apply_class} has unresolved required fields: " + ", ".join(unresolved_apply)
            )

        ip_apply_id = state.add_node(apply_class, apply_inputs)
        state.model_out = [ip_apply_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for(
            "CLIPVisionLoader",
            "IPAdapterModelLoader",
            "IPAdapterApply",
            "IPAdapterApplyAdvanced",
            "LoadImage",
        )


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(IpAdapterNode())
