from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .context import ComfyWorkflowBuildContext
from .contracts import WorkflowStageLabel


@dataclass(slots=True)
class ComfyWorkflowState:
    context: ComfyWorkflowBuildContext
    workflow: dict[str, Any] = field(default_factory=dict)
    next_id: int = 1
    labels_by_class_type: dict[str, WorkflowStageLabel] = field(default_factory=dict)

    request: Any = field(init=False)
    prompt: Any = field(init=False)
    models: Any = field(init=False)
    sampling: Any = field(init=False)
    image: Any = field(init=False)
    enhancements: Any = field(init=False)
    seed: int = field(init=False)

    model_out: list[Any] | None = None
    clip_out: list[Any] | None = None
    vae_out: list[Any] | None = None
    positive_ref: list[Any] | None = None
    negative_ref: list[Any] | None = None
    latent_id: str | None = None
    sampler_id: str | None = None
    image_out: list[Any] | None = None
    upscale_model_out: list[Any] | None = None
    base_width: int = 0
    base_height: int = 0

    def __post_init__(self) -> None:
        self.request = self.context.generation_request()
        self.prompt = self.request.prompt
        self.models = self.request.models
        self.sampling = self.request.sampling
        self.image = self.request.image
        self.enhancements = self.request.enhancements
        if self.sampling.seed >= 0:
            self.seed = self.sampling.seed
        else:
            self.seed = random.randint(0, 2**63 - 1)

    def add_node(self, class_type: str, inputs: dict[str, Any]) -> str:
        node_id = str(self.next_id)
        self.next_id += 1
        self.workflow[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    def set_workflow(self, workflow: dict[str, Any]) -> None:
        self.workflow = workflow
        self.next_id = len(workflow) + 1

    def add_stage_labels(self, labels: dict[str, WorkflowStageLabel]) -> None:
        self.labels_by_class_type.update(labels)
