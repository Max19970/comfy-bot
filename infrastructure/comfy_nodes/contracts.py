from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .state import ComfyWorkflowState


@dataclass(frozen=True, slots=True)
class WorkflowStageLabel:
    localization_key: str
    default_text: str


class ComfyWorkflowNode(Protocol):
    node_id: str
    phase: int
    order: int

    def apply(self, state: ComfyWorkflowState) -> None: ...

    def stage_labels(self) -> dict[str, WorkflowStageLabel]: ...
