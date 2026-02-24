from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class ControlNetNode:
    node_id = "controlnet"
    phase = 20
    order = 20

    def apply(self, state: ComfyWorkflowState) -> None:
        if not state.models.controlnet_name or not state.context.reference_image_name:
            return
        if state.positive_ref is None or state.negative_ref is None:
            raise RuntimeError("ControlNet node requires positive and negative conditioning")

        control_loader_inputs = state.context.required_input_defaults("ControlNetLoader")
        control_name_field = state.context.select_field_name(
            control_loader_inputs,
            ("control_net_name", "controlnet_name"),
        )
        if not control_name_field:
            return

        control_loader_inputs[control_name_field] = state.models.controlnet_name
        unresolved_control_loader = [
            name for name, value in control_loader_inputs.items() if value is None
        ]
        if unresolved_control_loader:
            raise RuntimeError(
                "ControlNetLoader has unresolved required fields: "
                + ", ".join(unresolved_control_loader)
            )

        control_loader_id = state.add_node("ControlNetLoader", control_loader_inputs)
        control_image_id = state.add_node(
            "LoadImage",
            {"image": state.context.reference_image_name},
        )

        apply_class = ""
        if "ControlNetApplyAdvanced" in state.context.object_info:
            apply_class = "ControlNetApplyAdvanced"
        elif "ControlNetApply" in state.context.object_info:
            apply_class = "ControlNetApply"
        if not apply_class:
            return

        control_apply_inputs = state.context.required_input_defaults(apply_class)
        pos_field = state.context.select_field_name(
            control_apply_inputs,
            ("positive", "conditioning"),
        )
        neg_field = state.context.select_field_name(control_apply_inputs, ("negative",))
        controlnet_field = state.context.select_field_name(
            control_apply_inputs,
            ("control_net", "controlnet"),
        )
        image_field = state.context.select_field_name(
            control_apply_inputs,
            ("image", "hint", "control_image"),
        )
        strength_field = state.context.select_field_name(control_apply_inputs, ("strength",))

        if not pos_field or not controlnet_field or not image_field:
            return

        control_apply_inputs[pos_field] = state.positive_ref
        if neg_field:
            control_apply_inputs[neg_field] = state.negative_ref
        control_apply_inputs[controlnet_field] = [control_loader_id, 0]
        control_apply_inputs[image_field] = [control_image_id, 0]
        if strength_field:
            control_apply_inputs[strength_field] = state.models.controlnet_strength

        for field_name, field_value in (
            ("start_percent", 0.0),
            ("end_percent", 1.0),
            ("start", 0.0),
            ("end", 1.0),
        ):
            if field_name in control_apply_inputs:
                control_apply_inputs[field_name] = field_value

        unresolved_control_apply = [
            name for name, value in control_apply_inputs.items() if value is None
        ]
        if unresolved_control_apply:
            raise RuntimeError(
                f"{apply_class} has unresolved required fields: "
                + ", ".join(unresolved_control_apply)
            )

        control_apply_id = state.add_node(apply_class, control_apply_inputs)
        state.positive_ref = [control_apply_id, 0]
        if neg_field:
            state.negative_ref = [control_apply_id, 1]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for(
            "ControlNetLoader",
            "ControlNetApply",
            "ControlNetApplyAdvanced",
            "LoadImage",
        )


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(ControlNetNode())
