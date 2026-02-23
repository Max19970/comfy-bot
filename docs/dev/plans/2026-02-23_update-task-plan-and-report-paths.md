# Task context

- User request: update the universal OpenCode instruction file to use new task artifact locations.
- Requested changes:
  - Plan storage path: `docs/dev/tasks/{short_task_name}`.
  - Stage report storage path: `docs/dev/tasks/{short_task_name}/reports`.
- Target file: `AGENTS.md`.

# Topic validation result

- Detected topics:
  - Update path requirements for plan artifacts.
  - Add explicit path requirements for stage reports.
- Validation outcome: one coherent topic (instruction-path policy update for task artifacts).
- Decision: pass Topic Validation Gate.

# Stage-by-stage plan

## Stage 1 - Update mandatory path policy in instructions

### Stage objective

- Replace current plan path requirement with the new task-based root path and short task folder.

### Sub-tasks

1. Edit `AGENTS.md` planning gate section to replace `docs/dev/plans` requirement with `docs/dev/tasks/{short_task_name}`.
2. Update naming guidance in the same section so it clearly binds plan location to task folder.
3. Keep wording strict and mandatory to preserve protocol behavior.

### Why this stage does not break project operability

- Change is documentation-only and affects future agent workflow rules, not runtime code.

### Validation/check steps

1. Read updated planning section and confirm no leftover requirement points to `docs/dev/plans` for task plans.
2. Confirm the path includes `{short_task_name}` exactly.

## Stage 2 - Add explicit stage report storage requirement

### Stage objective

- Define exact path where stage completion reports must be stored.

### Sub-tasks

1. Add requirement in `AGENTS.md` Stage Completion Report section that each stage report must be written to `docs/dev/tasks/{short_task_name}/reports`.
2. Specify minimum naming pattern for report files (stage index and concise slug) for predictable retrieval.
3. Ensure this new requirement does not conflict with existing approval and reporting gates.

### Why this stage does not break project operability

- Change is documentation-only and clarifies process; no executable behavior is altered.

### Validation/check steps

1. Read Stage Completion Report section and confirm exact path appears verbatim.
2. Verify no contradictory path instruction remains elsewhere in `AGENTS.md`.

## Stage 3 - Consistency pass and final verification

### Stage objective

- Ensure instruction consistency across the full protocol document.

### Sub-tasks

1. Run a full-text consistency pass on `AGENTS.md` for old path strings.
2. Harmonize terminology: "task folder", "plan artifact", and "stage report artifact".
3. Prepare stage completion summary with manual verification steps for user.

### Why this stage does not break project operability

- Purely textual consistency check in rules document.

### Validation/check steps

1. Confirm all path references are aligned to the new `docs/dev/tasks/{short_task_name}` structure.
2. Confirm report path is only `docs/dev/tasks/{short_task_name}/reports`.

# Stage acceptance criteria

- Stage 1 accepted when planning path in `AGENTS.md` is explicitly `docs/dev/tasks/{short_task_name}` and no legacy path remains for task plans.
- Stage 2 accepted when `AGENTS.md` explicitly requires stage reports to be stored under `docs/dev/tasks/{short_task_name}/reports`.
- Stage 3 accepted when no contradictory path references remain and wording is internally consistent.

# Risk notes

- Risk: accidental retention of legacy `docs/dev/plans` wording in other sections.
  - Mitigation: explicit full-text consistency pass.
- Risk: path placeholders interpreted ambiguously.
  - Mitigation: keep `{short_task_name}` literal and add concrete naming guidance.

# Unplanned additional tasks

- None.

# Stage approval log

- 2026-02-23: Plan approved by user.
- 2026-02-23: Stage 1 implemented; awaiting explicit user approval.
- 2026-02-23: Stage 1 approved by user.
- 2026-02-23: Stage 1 committed in `2df5088`.
- 2026-02-23: Stage 2 implemented; awaiting explicit user approval.
- 2026-02-23: Stage 2 approved by user.
- 2026-02-23: Stage 2 committed in `d11db19`.
- 2026-02-23: Stage 3 implemented; awaiting explicit user approval.
- 2026-02-23: Stage 3 approved by user.
- 2026-02-23: Stage 3 committed.
