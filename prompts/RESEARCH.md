# Research Mission

Continuously identify practical ways to improve this application.

## Priorities

1. Bugs and correctness
2. Security vulnerabilities
3. Performance bottlenecks
4. Reliability
5. Better architecture
6. Lower infrastructure/API/model cost
7. Improved AI model quality
8. Better user experience
9. New useful capabilities

## Research process

For each research cycle:

1. Inspect recent commits and changed files.
2. Review TODO/FIXME comments.
3. Review failing or flaky tests.
4. Inspect application logs and metrics if available.
5. Examine dependencies for useful new capabilities or known issues.
6. Research current approaches used by comparable projects.
7. Identify 1-5 improvements worth investigating.

For each finding, document:

- Problem
- Evidence
- Current behavior
- Proposed improvement
- Expected benefit
- Risk
- Implementation complexity
- Files/components affected
- Suggested tests
- Estimated priority

## Rules

Research first.

Do not modify the main application unless specifically permitted.

Experimental code must go in:

research/experiments/

Do not:
- deploy
- push to remote
- touch production credentials
- make purchases
- create paid resources
- delete production data

## Output

Update:

research/findings/YYYY-MM-DD.md

Maintain a ranked backlog in:

research/BACKLOG.md
