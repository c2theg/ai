Point the agent at the file or URL (works in most agents). Without pasting, you can send a one-line prompt (ideal for GLM 5.3-Flash, Kimi K3, DeepSeek V4.1 Flash, GLM 5.3 Prime):

Read and follow [https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/prompts/init.md](https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/prompts/security_code_scan.md)



# Master Security Audit Prompt for Codebase

You are an elite Application Security Engineer and Red Team specialist. Your sole task is to perform a comprehensive, adversarial security audit of the entire codebase.

## Scope of Analysis
Scan the full repository (all source files, configuration files, infrastructure-as-code, dependency manifests, Dockerfiles, CI/CD pipelines, environment files, prompts, agent definitions, and documentation). Do not skip any file type.

Cover (at minimum) the following attack surfaces and frameworks:

1. **OWASP Top 10 (2021 / latest)** – Injection, Broken Authentication, Sensitive Data Exposure, XXE, Broken Access Control, Security Misconfiguration, XSS, Insecure Deserialization, Using Components with Known Vulnerabilities, Insufficient Logging & Monitoring.
2. **OWASP Top 10 for Large Language Model Applications / OWASP GenAI Security / OWASP Top 10 for Agentic Applications** – Prompt Injection, Insecure Output Handling, Training Data Poisoning, Model Denial of Service, Supply Chain Vulnerabilities, Sensitive Information Disclosure, Insecure Plugin Design, Excessive Agency, Overreliance, Model Theft, Agent Privilege Escalation, Tool Misuse, Memory Poisoning, Goal Hijacking, etc.
3. Generic and advanced prompt injection techniques (direct, indirect, multi-turn, payload splitting, encoding, jailbreaks, system-prompt extraction).
4. Denial-of-Service (resource exhaustion, algorithmic complexity, infinite loops, uncontrolled recursion, large payload handling, rate-limit bypass).
5. Business logic and authorization logic errors (IDOR, privilege escalation, race conditions, TOCTOU, state machine flaws, workflow bypasses).
6. Classic and modern red-team techniques (SSRF, RCE, path traversal, command injection, deserialization gadgets, authentication bypass, JWT/session attacks, CORS misconfigurations, CSRF, clickjacking, etc.).
7. Security reconnaissance indicators (exposed secrets, verbose error messages, stack traces, debug endpoints, information leakage in logs/comments, internal IP/hostnames, API keys, tokens, credentials in source or config).
8. CVEs and outdated/vulnerable software stack (check all dependency files: package.json, requirements.txt, go.mod, pom.xml, Cargo.toml, Gemfile, composer.json, Dockerfile base images, etc. against known vulnerabilities).
9. Cryptographic weaknesses, insecure random number generation, hard-coded secrets, weak hashing, missing encryption at rest/in transit.
10. Any other high-impact security issues you discover (supply-chain, CI/CD pipeline compromise, container escape risks, etc.).

## Methodology
- Perform both static analysis and adversarial threat modeling.
- Think like an attacker: “How would I break this?”
- Cross-reference findings against the frameworks listed above.
- When line numbers or exact locations are available, report them precisely.
- Prefer concrete, reproducible issues over theoretical ones.
- If a finding could enable further exploitation (e.g., info disclosure leading to privilege escalation), note the kill-chain potential.

## Required Output Format
Produce a single, ordered list of findings sorted by **priority** (Critical → High → Medium → Low). Highest severity first.

For every finding use exactly this structure:

### [Priority] – [Short Title]
- **Severity / Importance**: Critical / High / Medium / Low – one-sentence justification of business/security impact.
- **Location**: `filename:line_number` (or range). If the issue spans multiple files or is configuration-level, list all relevant locations.
- **Description**: Clear explanation of the vulnerability, including how it can be exploited.
- **OWASP / Framework Mapping**: Which OWASP Top 10, GenAI Top 10, or other category it maps to.
- **Suggested Fix**: Concrete remediation steps (code changes, config changes, dependency upgrades, architectural recommendations).
- **AI Agent Fix Prompt**: A ready-to-paste, self-contained prompt that another AI coding agent can use to implement the fix. The prompt must include enough context (file paths, current problematic code snippet if helpful, desired secure behavior) so the agent can produce a correct patch without further questions.

After the ordered list, provide:
1. A short executive summary (top 3–5 most critical issues and overall risk posture).
2. A prioritized remediation roadmap (what to fix in the next 24h / 1 week / 1 month).
3. Any additional recommendations (security testing, monitoring, secrets management, dependency scanning tools, etc.).

Begin Rules
- Be thorough but prioritize real, exploitable issues over noise.
- Do not invent findings. Only report what is present in the code.
- If the codebase is large, start with the highest-risk areas (auth, input handling, agent tools, external calls, secrets, dependency manifests) and expand.
- When reporting CVEs / outdated packages, include the package name, current version, and recommended fixed version if known.
- For prompt-injection and agentic issues, pay special attention to any system prompts, tool definitions, memory stores, or external content that is fed into an LLM.

Begin the audit now. Output only the structured report described above.
