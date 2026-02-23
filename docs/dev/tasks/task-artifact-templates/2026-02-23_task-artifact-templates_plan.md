# Task context

- User request: create high-quality, attractive, and flexible templates for task plan files and stage report files.
- User request: update instructions so the agent must copy and fill the plan template when creating a plan, and copy and fill the stage report template when creating a stage report.
- Scope: process/instruction artifacts and markdown templates only.
- Primary target files (planned): `AGENTS.md` and new template files under `docs/dev/tasks`.
- Planned template output paths:
  - `docs/dev/tasks/_templates/task_plan_template.md`
  - `docs/dev/tasks/_templates/stage_completion_report_template.md`

# Topic validation result

- Detected topics:
  - Create reusable template artifacts for planning and stage reporting.
  - Enforce template-based workflow in the universal instructions.
- Validation outcome: one coherent topic (template-driven task artifact workflow).
- Decision: Topic Validation Gate passed.

# Stage-by-stage plan

## Stage 1 - Define template architecture and storage locations

### Stage objective

- Define stable, discoverable locations and naming conventions for reusable plan/report templates.

### Sub-tasks

1. Set canonical template directory path to `docs/dev/tasks/_templates`.
2. Set exact template filenames:
   - Task plan template: `docs/dev/tasks/_templates/task_plan_template.md`.
   - Stage completion report template: `docs/dev/tasks/_templates/stage_completion_report_template.md`.
3. Align template locations with existing instruction rules so runtime artifacts and templates do not conflict.

### Why this stage does not break overall project operability

- This stage only defines documentation structure and naming.
- No runtime code or application behavior is modified.

### Validation/check steps

1. Verify both template paths are unambiguous and future-proof.
2. Verify paths are consistent with the existing task artifact structure (`docs/dev/tasks/{short_task_name}` and `.../reports`).

## Stage 2 - Create high-quality flexible template files

### Stage objective

- Create polished markdown templates that are readable, attractive, and flexible for different task types.

### Sub-tasks

1. Create a task plan template with:
   - Clear mandatory sections.
   - Reusable placeholders.
   - Guidance for modular stages and validation criteria.
2. Create a stage completion report template with:
   - Clear section flow (implemented work, self-check results, manual verification).
   - Reusable placeholders for stage metadata.
   - Optional blocks for unplanned in-scope work.
3. Ensure template language supports different project contexts (feature, refactor, docs, bugfix).

### Why this stage does not break overall project operability

- This stage adds documentation templates only.
- No execution paths, dependencies, or builds are affected.

### Validation/check steps

1. Review template readability and structure.
2. Confirm placeholders are explicit and easy to replace.
3. Confirm templates can be used without additional edits to instructions.

## Stage 3 - Enforce template-copy workflow in `AGENTS.md`

### Stage objective

- Update protocol instructions so creating plans/reports requires copying and filling templates.

### Sub-tasks

1. Update Planning Gate rules to require:
   - Copying the plan template to the task plan artifact path.
   - Filling all required template sections.
2. Update Stage Completion Report rules to require:
   - Copying the stage report template to the stage report artifact path.
   - Filling all required sections and stage metadata.
3. Add explicit references to both template file paths in `AGENTS.md`.
4. Preserve strict gate behavior and approval requirements.

### Why this stage does not break overall project operability

- Changes are limited to process instructions.
- Existing code, tests, and runtime behavior remain unchanged.

### Validation/check steps

1. Verify `AGENTS.md` explicitly mandates copy+fill behavior for both artifact types.
2. Verify template paths are present and correct.
3. Verify no contradictory instruction remains.

## Stage 4 - Consistency pass and artifact integrity check

### Stage objective

- Ensure templates and instructions are fully aligned and internally consistent.

### Sub-tasks

1. Run consistency scan across `AGENTS.md` and the new template files.
2. Ensure terminology is consistent: template, plan artifact, stage report artifact, short task name.
3. Finalize documentation artifacts and prepare stage completion summary for user review.

### Why this stage does not break overall project operability

- Final verification stage only checks documentation consistency.

### Validation/check steps

1. Confirm required template files exist at the referenced paths.
2. Confirm all mandatory rule statements in `AGENTS.md` reflect template-driven flow.
3. Confirm no outdated path guidance remains.

# Stage acceptance criteria

- Stage 1 accepted when template directory and filenames are defined explicitly as `docs/dev/tasks/_templates/task_plan_template.md` and `docs/dev/tasks/_templates/stage_completion_report_template.md`, and aligned with existing task artifact structure.
- Stage 2 accepted when both templates exist, are polished, and include flexible placeholders/sections.
- Stage 3 accepted when `AGENTS.md` explicitly requires copying and filling templates for plan and stage report artifacts.
- Stage 4 accepted when all template references and path requirements are consistent across artifacts.

# Risk notes

- Risk: templates become too rigid for non-standard tasks.
  - Mitigation: include optional sections and clearly marked editable placeholders.
- Risk: instruction references drift from actual template paths.
  - Mitigation: add explicit path checks in validation steps.
- Risk: partial template filling leads to low-quality artifacts.
  - Mitigation: require completion of mandatory sections in instructions.

# Unplanned additional tasks

- Added `docs/dev/tasks/_templates/README.md` to make template directory purpose and usage contract explicit.
- Enhanced markdown presentation for both templates based on user follow-up feedback (denser structure, checklists, metadata tables, and clearer optional blocks).
- Performed an additional visual pass to further improve hierarchy and readability (navigation blocks, section legends, and snapshot tables).

# Stage approval log

- 2026-02-23: Plan updated to include explicit template file paths per user request.
- 2026-02-23: Plan approved by user.
- 2026-02-23: Stage 1 implemented; awaiting explicit user approval.
- 2026-02-23: Stage 1 approved by user.
- 2026-02-23: Stage 1 committed in `d8ea901`.
- 2026-02-23: Stage 2 implemented; awaiting explicit user approval.
- 2026-02-23: Stage 2 refined with markdown-formatting improvements requested by user; awaiting explicit user approval.
- 2026-02-23: Stage 2 refined again for stronger visual hierarchy and readability; awaiting explicit user approval.
- 2026-02-23: Stage 2 approved by user.
- 2026-02-23: Stage 2 committed in `40a0354`.
- 2026-02-23: Stage 3 implemented; awaiting explicit user approval.
- 2026-02-23: Stage 3 approved by user.
