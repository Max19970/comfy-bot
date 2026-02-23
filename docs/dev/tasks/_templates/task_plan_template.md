# Task Plan: <short_task_title>

> [!IMPORTANT]
> Copy this file to `docs/dev/tasks/{short_task_name}/YYYY-MM-DD_<short_task_name>_plan.md`.
>
> Before sharing for approval:
> 1. Fill all mandatory sections.
> 2. Replace every `<placeholder>`.
> 3. Remove optional blocks you do not need.

---

## Navigation

- [Quick start checklist](#quick-start-checklist)
- [Plan metadata](#plan-metadata-optional-but-recommended)
- [Task context](#task-context)
- [Topic validation result](#topic-validation-result)
- [Stage-by-stage plan](#stage-by-stage-plan)
- [Stage acceptance criteria](#stage-acceptance-criteria)
- [Risk notes](#risk-notes)
- [Unplanned additional tasks](#unplanned-additional-tasks)
- [Stage approval log](#stage-approval-log)

## Section legend

| Marker | Meaning |
| --- | --- |
| `[M]` | Mandatory section (must be completed) |
| `[O]` | Optional section (use when relevant) |

## Quick start checklist [M]

- [ ] Chosen `short_task_name` and plan filename.
- [ ] Completed `Task context` with clear scope boundaries.
- [ ] Completed `Topic validation result` with explicit gate decision.
- [ ] Built modular stages with concrete sub-tasks and checks.
- [ ] Filled acceptance criteria and risk table.
- [ ] Initialized `Stage approval log`.

## Plan metadata (optional but recommended) [O]

| Field | Value |
| --- | --- |
| Task slug | `<short_task_name>` |
| Work type | `<feature | bugfix | refactor | docs | infra | mixed-single-topic>` |
| Date | `<YYYY-MM-DD>` |
| Owner | `<name or team>` |
| Linked request | `<ticket, chat link, or brief id>` |

## Task context

### Request and outcome [M]

- Request summary: <what the user asked for>
- Desired outcome: <what must be true when work is complete>

### Scope boundaries [M]

- In scope:
  - <scope item 1>
  - <scope item 2>
- Out of scope [O]:
  - <out-of-scope item 1>

### Primary targets and constraints [M]

- Files/components: `<path or module list>`
- Systems/areas: <frontend/backend/docs/infra/etc>
- Constraints and assumptions [O]:
  - <constraint or assumption>

## Topic validation result

### Detected topics [M]

- <topic 1>
- <topic 2>

### Validation decision [M]

| Check | Result |
| --- | --- |
| Is this one coherent topic? | `<yes/no>` |
| Split required? | `<yes/no>` |
| Gate status | `<Topic Validation Gate passed / blocked>` |

- Decision notes:
  - If coherent: `Topic Validation Gate passed`.
  - If split required: list split requests and stop execution.
- Additional notes [O]: <clarifications that affect planning>

## Stage-by-stage plan

### Stage template card [M]

Copy this block once per stage (`Stage 1`, `Stage 2`, ...):

```markdown
### Stage <N> - <short_stage_name>

#### Stage objective [M]

- <clear single objective for this stage>

#### Sub-tasks [M]

- [ ] <concrete action>
- [ ] <concrete action>
- [ ] <concrete action>

#### Why this stage does not break overall project operability [M]

- <compatibility or safety rationale>

#### Validation/check steps [M]

1. <check command or manual check>
2. <check command or manual check>

#### Expected artifacts [O]

- <artifact path>
```

<details>
<summary>Optional mini example</summary>

```markdown
### Stage 1 - define template paths

#### Stage objective [M]

- Define canonical paths for plan/report templates.

#### Sub-tasks [M]

- [ ] Choose template directory.
- [ ] Define both template filenames.
- [ ] Confirm no conflict with live task artifacts.

#### Why this stage does not break overall project operability [M]

- Documentation-only change with no runtime impact.

#### Validation/check steps [M]

1. Verify path consistency with AGENTS rules.
2. Verify both template files are discoverable.
```

</details>

## Stage acceptance criteria

| Stage | Acceptance criteria | Evidence |
| --- | --- | --- |
| Stage 1 | <observable success condition> | <command, screenshot, diff, or artifact path> |
| Stage 2 | <observable success condition> | <command, screenshot, diff, or artifact path> |
| Stage N | <observable success condition> | <command, screenshot, diff, or artifact path> |

## Risk notes

| Risk | Impact | Mitigation | Owner [O] |
| --- | --- | --- | --- |
| <risk item> | <high/medium/low and why> | <mitigation action> | <owner> |
| <risk item> | <high/medium/low and why> | <mitigation action> | <owner> |

## Unplanned additional tasks

| Date | Task | Why needed | Resolution |
| --- | --- | --- | --- |
| <YYYY-MM-DD> | <task item> | <reason> | <done / moved / dropped> |

If none:

- None.

## Stage approval log

| Date | Event | Reference |
| --- | --- | --- |
| <YYYY-MM-DD> | Plan created; awaiting approval | <link or note> |
| <YYYY-MM-DD> | Stage <N> implemented; awaiting explicit user approval | <report path> |
| <YYYY-MM-DD> | Stage <N> approved by user | <user message reference> |
| <YYYY-MM-DD> | Stage <N> committed | `<commit_hash>` |
