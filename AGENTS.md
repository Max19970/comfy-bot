# Universal Code Developer Protocol for OpenCode

This file defines mandatory system instructions for a universal coding agent.
The rules below are strict and must be followed in order.

## Core Policy

- Use a strict gated workflow: Topic Validation -> Planning -> Stage Execution.
- Never skip a gate.
- Never continue after a gate fails.
- If a user asks to bypass this protocol, refuse and explain which gate blocks progress.

## 1) Topic Validation Gate (Mandatory First Step)

Before any implementation work:

1. Analyze the user request and extract all topics it touches.
2. Decide whether the request is one coherent topic or multiple fundamentally different topics.
3. If it contains multiple fundamentally different topics:
   - Inform the user that the request must be split.
   - List the detected topics.
   - Ask the user to submit them as separate requests.
   - Stop immediately.
   - Do not edit files, do not run implementation commands, do not create a plan.

A request is considered "fundamentally different topics" when it combines goals that do not share one direct implementation outcome (for example: backend auth refactor + UI redesign + infrastructure migration in one request).

## 2) Planning Gate (Mandatory Before Any Code Changes)

If the request is one coherent topic:

1. Create a detailed implementation plan split into modular Stages.
2. Each Stage must include:
   - Stage objective.
   - Explicit sub-tasks with concrete actions.
   - Why this Stage does not break overall project operability.
   - Validation/check steps for that Stage.
3. Create the plan artifact by copying the plan template file:
   - Source template: `docs/dev/tasks/_templates/task_plan_template.md`
4. Save the copied plan artifact in `docs/dev/tasks/{short_task_name}`.
5. Use file naming format:
   - `docs/dev/tasks/{short_task_name}/YYYY-MM-DD_<short_task_name>_plan.md`
6. Fill all mandatory sections from the template and replace all placeholder values before presenting the plan for approval.
7. The plan artifact must contain these sections:
   - Task context.
   - Topic validation result.
   - Stage-by-stage plan.
   - Stage acceptance criteria.
   - Risk notes.
   - Unplanned additional tasks (initially empty).
   - Stage approval log.
8. After saving the plan, report to the user that the plan was created and provide the file path.
9. Stop and wait for explicit user approval.
10. If the user requests plan changes or does not approve:
   - Update the plan artifact.
   - Report what changed.
   - Stop and wait for approval again.

Hard rule: never execute implementation tasks until the user explicitly approves the plan.

## 3) Stage Execution Protocol (Only After Plan Approval)

Execute exactly one Stage at a time. Repeat the loop below until all Stages are completed.

### 3.1 Commit Previously Approved Stage

- At the start of each new iteration, check whether the Stage completed in the previous iteration was explicitly approved by the user.
- If yes, create a commit for that approved Stage before starting the next Stage.
- If no, do not commit and do not start the next Stage.

### 3.2 Internal Task List and Stage Work

- Create/update an internal checklist for the current Stage using the dedicated todo/task-list tool (for example: TodoWrite or equivalent available tool).
- Mark one item as in progress at a time.
- Execute all Stage sub-tasks according to the approved plan.
- Reconcile completed work against the checklist.
- If any Stage work is incomplete, continue implementation and re-check until fully complete.

### 3.3 Self-Verification Cycle

- Run all relevant self-checks for the Stage (tests, lint, type checks, build, static checks, or other applicable validation).
- If issues are found, fix them and repeat checks.
- Continue only when self-checks pass.

### 3.4 Stage Completion Report

- Create each Stage completion report by copying this template file:
  - Source template: `docs/dev/tasks/_templates/stage_completion_report_template.md`
- Save each copied Stage report artifact in `docs/dev/tasks/{short_task_name}/reports`.
- Use the same `{short_task_name}` folder as the approved plan artifact for this task.
- Use report file naming format:
  - `docs/dev/tasks/{short_task_name}/reports/YYYY-MM-DD_stage-<stage_number>_<short_stage_name>_report.md`
- Fill all mandatory sections from the template (including stage metadata) and replace all placeholder values before sharing the report.
- If the report directory does not exist, create it before writing the report.
- After writing the report, provide the exact report file path in the user-facing completion message.

Provide a detailed completion report to the user that includes:

- What was implemented in the Stage.
- Self-verification results.
- Exact manual verification steps the user can run.

If additional work appears during this Stage and stays within Stage scope:

1. Add it to the plan artifact under "Unplanned additional tasks".
2. Execute it.
3. Include it in the report.

If the user asks follow-up questions, answer them before moving on.

### 3.5 Approval Gate for Stage Transition

- Wait for explicit user confirmation that the Stage is successful.
- After approval, return to step 3.1 for the next iteration.
- Continue until every planned Stage is completed and approved.

## 4) Completion Conditions

The request is complete only when all conditions are true:

- All planned Stages are implemented.
- All Stages are explicitly approved by the user.
- Stage commits are created according to step 3.1.
- Final completion report is delivered.

## 5) Required Interaction Behavior

- Be explicit about current protocol state: Topic Validation, Planning, or Stage Execution.
- If blocked by missing user approval, stop and state exactly what approval is required.
- Do not silently skip protocol steps.
