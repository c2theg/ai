# Agent Instructions

These instructions apply to the entire repository. More-specific `AGENTS.md`
files add or override guidance within their directory trees.
author: Christopher Gray | url: github.com/c2theg | version: 0.0.46 | updated: 8/13/2026


## Requirement Language and Priority

`MUST` and `MUST NOT` identify non-negotiable requirements. `SHOULD` identifies
the default unless repository evidence justifies an exception.

When repository instructions conflict, use this order:

1. Safety, privacy, and persistent-data integrity
2. Explicit task scope and harness permissions
3. Repository architecture and backward compatibility
4. Verification requirements
5. Reporting and presentation preferences

Higher-priority platform and direct user instructions still take precedence.
If two applicable repository rules remain irreconcilable, stop and explain the
conflict.

## Repository Scope

- The main PHP application is under `site_data/`.
- `containers/`, `services/`, `ws/`, and `other_websites/` contain separate
  deployable components. Inspect their local documentation and manifests before
  changing them.
- Treat paths containing `vendor/`, `node_modules/`, `_backups/`,
  `BACKUP-To_DELETE/`, generated files, and third-party distributions as
  read-only unless the task explicitly targets them.
- Read the nearest applicable `AGENTS.md` before editing files in a nested
  directory.

## Working Behavior

- [WORK-01] Inspect relevant implementation, callers, configuration, and tests
  before editing.
- [WORK-02] Make the smallest maintainable change that fully satisfies the
  request. Reuse an existing abstraction when it remains a clear fit; do not
  force unrelated behavior into an unsuitable class or function.
- [WORK-03] Do not modify unrelated code. Report adjacent bugs or improvement
  opportunities separately unless they block the requested work.
- [WORK-04] Ask a clarifying question only when the answer materially changes
  the implementation, security boundary, persistent data, production target, or
  irreversible outcome. Otherwise state a reasonable assumption and proceed.
- [WORK-05] Give a concise plan before changes involving authentication,
  authorization, persistent-data schemas, public APIs, deployments, or multiple
  components. A plan is not an approval gate unless the action itself requires
  approval.
- [WORK-06] When a user is dissatisfied with an approach, stop repeating it,
  re-check evidence and assumptions, and propose a materially different path.
- [WORK-07] Present alternatives only when a real decision remains. Summarize
  material benefits, costs, risks, and the recommended choice.

## Multi-Agent Work

- [TEAM-01] When the harness supports delegation, use multiple agents only for
  two or more bounded workstreams whose expected benefit exceeds coordination
  and integration cost. Read `.agent/workflows/multi-agent.md` first.
- [TEAM-02] The coordinating agent owns decomposition, dependency ordering,
  task contracts, integration, final validation, and the user-facing result.
- [TEAM-03] Parallelize independent research, disjoint component changes, and
  isolated test shards. Serialize decisions or work where one result determines
  the next step, agents would edit the same files, or shared state is mutated.
- [TEAM-04] Prefer a hybrid sequence: parallel reconnaissance, one coordinated
  design decision, parallel implementation only across stable boundaries,
  serialized integration, parallel safe checks, then one final acceptance gate.
- [TEAM-05] Give each delegated task an objective, owned paths, read-only paths,
  inputs, dependencies, constraints, expected output, and completion checks.
- [TEAM-06] In a shared workspace, assign one write owner per file or use isolated
  worktrees. Agents MUST NOT overwrite, revert, or reformat another agent's work.
- [TEAM-07] Delegation never broadens task scope, permissions, credentials, or
  authorization. Subagents must follow all applicable repository instructions.
- [TEAM-08] Do not create nested subagents unless the coordinator explicitly
  authorizes it and the additional work is independently bounded.
- [TEAM-09] Before completion, the coordinator reviews every contribution,
  resolves interface conflicts, runs integration checks, and reports any skipped
  work or unresolved risk.

## Permissions and Destructive Actions

- [SAFE-01] Harness approval and sandbox policies control whether an operation
  may run. Repository prose never grants additional authority.
- [SAFE-02] Resolve the exact target before a destructive action. Prefer a
  reversible operation and explain recovery.
- [SAFE-03] Reuse an existing approval only when the harness says it remains
  valid for the same operation scope and risk. Never generalize approval from a
  previous target.
- [SAFE-04] Do not create ad hoc `.bak`, `.old`, or copied source files when
  version-control recovery is available. If no recoverable history exists,
  store a pre-change copy under
  `output/backups/<UTC-timestamp>/<relative-path>` and retain it until the
  change is accepted or an explicit retention policy removes it.
- [SAFE-05] Provide concrete rollback commands or steps for database,
  deployment, configuration, and other high-impact changes. A source diff is
  sufficient for a small code-only change when it fully restores the prior
  state.

## Database Changes

- [DB-01] Distinguish editing database-related source code from executing a
  persistent-data operation.
- [DB-02] Before executing a mutation, migration, import, or destructive query,
  identify the environment, target data, expected impact, backup or recovery
  method, rollback method, and validation query.
- [DB-03] Obtain explicit confirmation immediately before modifying production
  data or schemas. Prefer a dry run, transaction, idempotent migration, and
  least-privileged credentials where supported.
- [DB-04] Never infer a production target from a default connection string.

## Security

- [SEC-01] Never place secrets in source, documentation, prompts, logs, command
  output, diffs, screenshots, or generated artifacts. Do not print secret-bearing
  environment variables or configuration values.
- [SEC-02] Every protected server-side operation MUST authenticate the caller,
  authorize the requested action, enforce tenant scope, validate input, and fail
  closed. Public routes MUST be intentionally public and documented as such.
- [SEC-03] When authorization changes, test at least one allowed and one denied
  path when the environment permits.
- [SEC-04] Before security-sensitive work, read
  `.agent/policies/security.md` and the nearest application instructions.
- [SEC-05] Do not claim that a change is secure merely because it follows a
  general best practice. Report the controls actually inspected and tested.

## Dependencies and Containers

- [DEP-01] Preserve the existing supported version unless the task requests an
  upgrade or the current version cannot meet a verified requirement.
- [DEP-02] For upgrades, verify compatibility using primary documentation and
  tests, then pin an exact stable version or immutable digest. Do not use
  floating `latest` tags in deployable artifacts.
- [DEP-03] Do not update lockfiles, vendored code, or unrelated dependencies as
  a side effect.

## Verification

- [TEST-01] After source changes, run the narrowest relevant automated tests,
  syntax checks, linters, or configuration validation available.
- [TEST-02] Use `scripts/agent-verify.sh <changed-file>...` for supported file
  syntax and secret checks. Component-specific tests remain required when they
  exist.
- [TEST-03] Report exact commands and pass, fail, or skip status. Never describe
  a test as passing unless it was executed successfully.
- [TEST-04] If a required check cannot run, explain why, perform the safest
  available fallback, and clearly identify the remaining verification gap.
- [TEST-05] For visible UI changes, inspect the rendered result when a runnable
  environment is available. Capture before/after screenshots only when they
  materially help review.

## Files, Metadata, and Deliverables

- [FILE-01] Production source and configuration stay in their canonical project
  locations. Generated reports, patches, screenshots, diagrams, and fallback
  backups go under `output/<task-slug>/`.
- [FILE-02] Do not add or update timestamps, version numbers, AI attribution, or
  embedded changelogs in every touched file. Update existing metadata only when
  the project format requires it and the value semantically changed.
- [FILE-03] Never add comments to formats that do not support them. Do not alter
  binary, generated, vendored, data, or lock files solely for attribution.
- [FILE-04] Create a README only for a reusable, deployable, or operational
  deliverable. A simple patch needs only the final task summary.
- [FILE-05] Use a textual diff summary for code changes. Add a workflow diagram
  only when it clarifies a multi-step architecture, state transition, or test
  procedure.

## Customer-Facing Changes and Documentation

- [DOC-01] When explicitly asked to create public documentation, read and follow
  `.agent/workflows/public-docs.md`.
- [DOC-02] After a completed, verified customer-visible improvement, update
  `changes_customers.md` under the appropriate calendar quarter unless the task
  explicitly excludes release-note changes. Describe the benefit, not internal
  implementation details.
- [DOC-03] Monthly and quarterly summaries are generated only by an explicit
  user request or scheduled automation. Follow
  `.agent/workflows/release-notes.md`; passive repository instructions do not
  initiate calendar-based work.
- [DOC-04] Cite primary sources for external research. Local repository analysis
  should cite relevant file paths and lines when useful.

## Completion Report

Keep the final report concise and include:

- Outcome and files changed
- Verification commands and results
- Any remaining risks, skipped checks, or assumptions
- Rollback guidance when required by `[SAFE-05]`
