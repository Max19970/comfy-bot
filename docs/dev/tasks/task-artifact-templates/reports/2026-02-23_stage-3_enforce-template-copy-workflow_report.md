# Stage 3 completion report: enforce-template-copy-workflow

## Stage metadata

| Field | Value |
| --- | --- |
| Task | `task-artifact-templates` |
| Work type | `docs` |
| Stage number | `3` |
| Stage short name | `enforce-template-copy-workflow` |
| Date | `2026-02-23` |
| Related plan | `docs/dev/tasks/task-artifact-templates/2026-02-23_task-artifact-templates_plan.md` |

## What was implemented

- Updated `AGENTS.md` Planning Gate to require copying the plan template:
  - Source template: `docs/dev/tasks/_templates/task_plan_template.md`
- Updated `AGENTS.md` Stage Completion Report protocol to require copying the stage report template:
  - Source template: `docs/dev/tasks/_templates/stage_completion_report_template.md`
- Added explicit instruction that mandatory sections must be filled and placeholders replaced for both plan and report artifacts.
- Preserved existing gating and approval logic while introducing template-copy workflow requirements.

## Self-verification results

### Verification snapshot

| Metric | Value |
| --- | --- |
| Total checks | `4` |
| Passed | `4` |
| Failed | `0` |
| Final status | `pass` |

### Check summary

| Check | Command or method | Result | Notes |
| --- | --- | --- | --- |
| Plan template path present in AGENTS | `grep for task_plan_template.md` | pass | Found explicit source template line in Planning Gate |
| Stage report template path present in AGENTS | `grep for stage_completion_report_template.md` | pass | Found explicit source template line in Stage Completion Report section |
| Copy+fill requirements for plans present | `grep for copying/fill mandatory/replace placeholders` | pass | Found explicit copy and fill statements |
| Copy+fill requirements for reports present | `grep for copying/fill mandatory/replace placeholders` | pass | Found explicit copy and fill statements |

## Manual verification steps

1. Open `AGENTS.md` and verify the Planning Gate references `docs/dev/tasks/_templates/task_plan_template.md` as a source template.
2. Open `AGENTS.md` and verify the Stage Completion Report section references `docs/dev/tasks/_templates/stage_completion_report_template.md` as a source template.
3. Confirm `AGENTS.md` includes explicit copy+fill requirements for both artifacts (mandatory sections + placeholder replacement).

## Changed artifacts

| Artifact path | Change type | Purpose |
| --- | --- | --- |
| `AGENTS.md` | `updated` | Enforce template-copy workflow for plan and stage report creation |
| `docs/dev/tasks/task-artifact-templates/2026-02-23_task-artifact-templates_plan.md` | `updated` | Record Stage 2 commit and Stage 3 completion status |
| `docs/dev/tasks/task-artifact-templates/reports/2026-02-23_stage-3_enforce-template-copy-workflow_report.md` | `added` | Stage 3 completion report artifact |

## Approval request

| Item | Value |
| --- | --- |
| Status | `Awaiting explicit user approval for Stage 3` |
| Next action after approval | commit Stage 3 and proceed to Stage 4 consistency pass |
