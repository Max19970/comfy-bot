# Task Plan: agent-work-modes

> [!IMPORTANT]
> This plan is created from `docs/dev/tasks/_templates/task_plan_template.md`.
>
> Before implementation starts:
> 1. Plan must be explicitly approved by user.
> 2. All stages are executed one by one.
> 3. Each completed stage requires explicit user approval before moving to the next stage.

## Quick start checklist [M]

- [x] Chosen `short_task_name` and plan filename.
- [x] Completed `Task context` with clear scope boundaries.
- [x] Completed `Topic validation result` with explicit gate decision.
- [x] Built modular stages with concrete sub-tasks and checks.
- [x] Filled acceptance criteria and risk table.
- [x] Initialized `Stage approval log`.

## Plan metadata (optional but recommended) [O]

| Field | Value |
| --- | --- |
| Task slug | `agent-work-modes` |
| Work type | `docs` |
| Date | `2026-02-23` |
| Owner | `OpenCode agent` |
| Linked request | `Add Simple/Extended execution modes to AGENTS.md` |

## Task context

### Request and outcome [M]

- Request summary: update `AGENTS.md` so agent supports two user-selectable modes: `Simple` and `Extended`.
- Desired outcome:
  - If user requests `Simple` mode: agent executes directly in a fast, practical format without mandatory expanded protocol overhead, while avoiding harmful shortcuts for project quality.
  - If user requests `Extended` mode: current strict gated protocol in `AGENTS.md` remains in effect.

### Scope boundaries [M]

- In scope:
  - Update instruction policy in `AGENTS.md` to define and enforce both modes.
  - Preserve existing expanded workflow semantics as the `Extended` mode baseline.
  - Define mode-detection behavior (how agent decides which mode to use from user request).
- Out of scope [O]:
  - Changes to code/runtime logic outside instruction documents.
  - Creation of new external config files unless strictly required.

### Primary targets and constraints [M]

- Files/components: `AGENTS.md`
- Systems/areas: process documentation / agent behavior policy
- Constraints and assumptions [O]:
  - Mode name in user prompt should match instruction language (`Simple` / `Extended`).
  - Default behavior when mode is not specified must be clearly documented.

## Topic validation result

### Detected topics [M]

- Define dual execution modes for the agent.
- Adjust instruction semantics to conditionally apply strict protocol only in extended mode.

### Validation decision [M]

| Check | Result |
| --- | --- |
| Is this one coherent topic? | `yes` |
| Split required? | `no` |
| Gate status | `Topic Validation Gate passed` |

- Decision notes:
  - Request is a single coherent topic: behavior policy extension in `AGENTS.md`.

## Stage-by-stage plan

### Stage 1 - Define mode policy model

#### Stage objective [M]

- Define exact semantics of `Simple` vs `Extended` modes, including activation, defaults, and safety boundaries.

#### Sub-tasks [M]

- [ ] Specify how mode is detected from user request text.
- [ ] Specify fallback/default behavior when user does not provide mode.
- [ ] Define constraints for `Simple` mode (fast path + no harmful shortcuts).
- [ ] Define that `Extended` mode uses the current strict workflow.

#### Why this stage does not break overall project operability [M]

- This is a documentation-only policy definition; no executable project code is changed.

#### Validation/check steps [M]

1. Re-read Stage 1 edits in `AGENTS.md` and verify both modes are unambiguous.
2. Verify no contradiction with existing safety/hygiene rules.

#### Expected artifacts [O]

- `AGENTS.md`

### Stage 2 - Integrate mode-aware workflow gates into AGENTS

#### Stage objective [M]

- Update gate rules so expanded protocol remains mandatory only for `Extended` mode, while `Simple` mode allows direct execution style.

#### Sub-tasks [M]

- [ ] Add a dedicated section describing both modes and precedence.
- [ ] Scope Topic Validation/Planning/Stage Execution gates to `Extended` mode.
- [ ] Add explicit operating behavior for `Simple` mode (fast execution, concise reporting, maintain project quality).
- [ ] Ensure language is strict enough to prevent accidental mixing of modes.

#### Why this stage does not break overall project operability [M]

- Change is isolated to agent instruction document; project runtime remains unaffected.

#### Validation/check steps [M]

1. Verify all gate sections explicitly reference `Extended` mode applicability.
2. Verify `Simple` mode has clear behavior and safety constraints.
3. Verify no section still implies expanded flow is always mandatory.

#### Expected artifacts [O]

- `AGENTS.md`

### Stage 3 - Consistency pass and completion artifacts

#### Stage objective [M]

- Ensure internal consistency and provide final stage artifact reporting under current extended-session protocol.

#### Sub-tasks [M]

- [ ] Run consistency scan in `AGENTS.md` for mode naming and rule collisions.
- [ ] Update plan `Unplanned additional tasks` if any within-scope additions appear.
- [ ] Create stage completion report artifact for this stage.

#### Why this stage does not break overall project operability [M]

- Final documentation verification only; no runtime effect.

#### Validation/check steps [M]

1. Confirm both mode names appear consistently (`Simple`, `Extended`).
2. Confirm expanded protocol instructions remain intact for `Extended` mode.
3. Confirm simple-mode instructions do not permit dangerous/low-quality shortcuts.

#### Expected artifacts [O]

- `AGENTS.md`
- `docs/dev/tasks/agent-work-modes/reports/YYYY-MM-DD_stage-N_short-stage-name_report.md`

## Stage acceptance criteria

| Stage | Acceptance criteria | Evidence |
| --- | --- | --- |
| Stage 1 | Mode semantics, activation, and defaults are explicitly defined | `AGENTS.md` diff review |
| Stage 2 | AGENTS gates are clearly scoped to `Extended` mode; `Simple` behavior documented | `AGENTS.md` diff review |
| Stage 3 | Full consistency check passes; no contradictions remain | grep/read checks + stage report artifact |

## Risk notes

| Risk | Impact | Mitigation | Owner [O] |
| --- | --- | --- | --- |
| Ambiguous mode selection when user omits mode | medium, inconsistent workflow behavior | Define explicit default mode in AGENTS | OpenCode agent |
| Simple mode interpreted as permissive for hacks | high, project quality risk | Add strict quality guardrail language for simple mode | OpenCode agent |
| Expanded-mode gates accidentally weakened | high, protocol regression | Scope changes carefully; verify all original gates preserved for `Extended` | OpenCode agent |

## Unplanned additional tasks

| Date | Task | Why needed | Resolution |
| --- | --- | --- | --- |
| 2026-02-23 | Remove residual placeholder rows/tokens in plan artifact | Keep artifact compliant with template rule requiring placeholder replacement before approval/workflow use | done |
| 2026-02-23 | Localize mode names to instruction language in AGENTS and stage artifacts | User follow-up requested localization consistency with instruction language | done |

If none:

- N/A (one additional task was executed and logged above).

## Stage approval log

| Date | Event | Reference |
| --- | --- | --- |
| 2026-02-23 | Plan created; awaiting approval | `docs/dev/tasks/agent-work-modes/2026-02-23_agent-work-modes_plan.md` |
| 2026-02-23 | Plan approved by user | User message: "План одобряется" |
| 2026-02-23 | Stage 1 implemented; awaiting explicit user approval | `docs/dev/tasks/agent-work-modes/reports/2026-02-23_stage-1_define-mode-policy-model_report.md` |
| 2026-02-23 | Stage 1 updated after localization follow-up; awaiting explicit user approval | `docs/dev/tasks/agent-work-modes/reports/2026-02-23_stage-1_define-mode-policy-model_report.md` |
| 2026-02-23 | Stage 1 approved by user | User message: "Этап 1 одобрен" |
