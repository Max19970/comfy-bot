# Stage <stage_number> completion report: <short_stage_name>

> [!IMPORTANT]
> Copy this file to `docs/dev/tasks/{short_task_name}/reports/YYYY-MM-DD_stage-<stage_number>_<short_stage_name>_report.md`.
>
> Before requesting approval:
> 1. Fill all mandatory sections.
> 2. Include exact file paths in `Changed artifacts`.
> 3. Replace all `<placeholder>` values.

---

## Navigation

- [Quick report checklist](#quick-report-checklist)
- [Stage metadata](#stage-metadata)
- [What was implemented](#what-was-implemented)
- [Self-verification results](#self-verification-results)
- [Manual verification steps](#manual-verification-steps)
- [Changed artifacts](#changed-artifacts)
- [Approval request](#approval-request)

## Section legend

| Marker | Meaning |
| --- | --- |
| `[M]` | Mandatory section |
| `[O]` | Optional section |

## Quick report checklist [M]

- [ ] Filled stage metadata and related plan path.
- [ ] Listed concrete implemented changes.
- [ ] Added self-verification outcomes with clear pass/fail status.
- [ ] Added exact manual verification steps.
- [ ] Listed changed artifact paths.
- [ ] Requested explicit approval.

## Stage metadata

| Field | Value |
| --- | --- |
| Task | `<short_task_name>` |
| Work type | `<feature | bugfix | refactor | docs | infra | mixed-single-topic>` |
| Stage number | `<number>` |
| Stage short name | `<short_stage_name>` |
| Date | `<YYYY-MM-DD>` |
| Related plan | `docs/dev/tasks/<short_task_name>/YYYY-MM-DD_<short_task_name>_plan.md` |

## What was implemented

- <implemented item 1>
- <implemented item 2>
- <implemented item 3>

## Self-verification results

### Verification snapshot [M]

| Metric | Value |
| --- | --- |
| Total checks | `<number>` |
| Passed | `<number>` |
| Failed | `<number>` |
| Final status | `<pass/fail>` |

### Check summary [M]

| Check | Command or method | Result | Notes |
| --- | --- | --- | --- |
| <lint/test/type/build/manual check> | `<command or method>` | <pass/fail> | <key output or observation> |
| <lint/test/type/build/manual check> | `<command or method>` | <pass/fail> | <key output or observation> |

### Defects found and resolved [O]

| Defect | Fix applied | Re-check result |
| --- | --- | --- |
| <issue description> | <fix summary> | <pass/fail and evidence> |

If none:

- No defects found during this stage.

## Manual verification steps

1. <step user can run manually>
2. <step user can run manually>
3. <expected observable outcome>

## Changed artifacts

| Artifact path | Change type | Purpose |
| --- | --- | --- |
| `<path>` | <added/updated/deleted> | <short rationale> |
| `<path>` | <added/updated/deleted> | <short rationale> |

## Unplanned additional tasks executed [O]

| Task | Why needed | Plan log reference | Outcome |
| --- | --- | --- | --- |
| <extra in-scope task> | <reason> | `<line or entry reference>` | <result> |

## Known limitations or follow-up notes [O]

- <limitation, caveat, dependency, or follow-up note>

## Approval request

| Item | Value |
| --- | --- |
| Status | `Awaiting explicit user approval for Stage <stage_number>` |
| Next action after approval | <commit this stage / move to next stage> |
