from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .context import ComfyWorkflowBuildContext
from .contracts import WorkflowStageLabel


@dataclass(slots=True)
class ComfyWorkflowState:
    context: ComfyWorkflowBuildContext
    workflow: dict[str, Any] = field(default_factory=dict)
    labels_by_class_type: dict[str, WorkflowStageLabel] = field(default_factory=dict)

    def set_workflow(self, workflow: dict[str, Any]) -> None:
        self.workflow = workflow

    def add_stage_labels(self, labels: dict[str, WorkflowStageLabel]) -> None:
        self.labels_by_class_type.update(labels)
