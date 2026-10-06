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

## Recursive Self-Improvement

The research system should also evaluate its own effectiveness.

During each research cycle, examine:

- prompt quality
- model selection
- tool selection
- retrieval quality
- context efficiency
- token consumption
- latency
- research accuracy
- false-positive recommendations
- benchmark coverage
- experiment success rate

The agent may propose improvements to:

- AGENTS.md
- RESEARCH.md
- research prompts
- model routing
- RAG configuration
- embeddings
- tool usage
- benchmark suites
- evaluation methodology
- speculative decoding configuration

Self-improvements must be evaluated exactly like application changes.

The agent MUST NOT automatically replace its own production instructions.

All recursive improvements go through:

PROPOSE
→ EXPERIMENT
→ BENCHMARK
→ REVIEW
→ APPROVE
→ MERGE


Generation N

Researcher
   ↓
"I think Researcher v2 would work better"

Experiment branch
   ↓
Researcher v2

        A/B TEST
       /        \
 Researcher v1   Researcher v2
       \          /
        Evaluator

Metrics:
• useful findings
• false positives
• code quality
• tests passed
• latency
• token cost
• security
• benchmark improvements

        ↓

v2 wins by required threshold?
          │
       yes/no


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
