# Master Prompt: Host-Local Logging + Rotation for Dockerized Services

**What it does:** A reusable, copy-paste prompt that has an AI assistant apply the
same logging pattern built for `example_app` to any other Dockerized service —
host-local log storage (not inside a synced/source-controlled project dir) plus a
logrotate sidecar with a real, enforced size cap.

**Why this exists:** Two production incidents on `example_app` both traced back to the
same two root causes: (1) logs stored in a directory that a sync tool (Syncthing) could
flip to world-writable, silently disabling logrotate's paranoia checks, and (2) a
logrotate sidecar that only checked file sizes once every 24 hours, so `maxsize` was
aspirational rather than enforced. Result: `audit.log` hit 1.3GB, later another
log hit 3GB. This prompt bakes in the fix so the next container doesn't repeat it.

**Intended audience:** Whoever (human or AI) is standing up or auditing logging for a
Docker Compose service — this project or another one.

**How to use:** Copy the prompt block below into a new session, fill in the
`<PARAMETERS>` block for the target service, and hand it to Claude Code (or follow it
yourself as a checklist).

**Security note:** The pattern uses `chmod 777` on the host log directory and
`user: root` inside the writing container — that's intentional (multiple
containers/UIDs need to write/read the same bind mount) but means anything else with
host filesystem access can read/tamper with logs. Keep the log directory outside any
web-served or shared path, and don't reuse it for anything containing secrets.

**Rollback:** Every step below is additive or copies (not moves) data until a final,
explicit cleanup step — see "Migration & rollback" in the prompt. Nothing destructive
happens until you've verified the new setup works.

---

## The prompt

```
Apply host-local logging + enforced-size log rotation to the Docker Compose service(s)
below, following the pattern already proven on the example_app project.

<PARAMETERS — fill in before running>
  SERVICE_NAME(S):        e.g. my-api, my-worker   (docker-compose service keys)
  CONTAINER_NAME(S):      e.g. my_api_container    (matching `container_name:` values)
  LOG_HOST_DIR:           e.g. /var/log/my-api      (host path — see constraints below)
  LOG_FILE_PATTERN(S):    e.g. access*.log error*.log app*.log
  FD_HELD_OPEN_LOGS:      any log file(s) written by a process that does NOT release
                           its file descriptor on a rename/reopen signal (e.g. an
                           embedded WAF/audit module, a language runtime with its own
                           logger) — list them, or write "none"
  MAX_SIZE_MB:            default 150 unless told otherwise
  BACKUP_COUNT:            default 1 (single compressed backup) unless told otherwise
  CHECK_INTERVAL:         default 1h — see "why hourly" below before changing
  READ_ONLY_CONSUMERS:    other containers that need read access to these logs
                           (e.g. an IDS, a log dashboard) — or "none"
</PARAMETERS>

CONSTRAINTS (do not violate these — they are root causes of prior incidents):

1. LOG_HOST_DIR must NOT be inside a directory that's under version control, or
   continuously synced by a tool like Syncthing/Dropbox/rsync-daemon/OneDrive. Those
   tools can flip directory permissions to world-writable on their own schedule,
   which trips logrotate's "insecure permissions" safety check and causes it to
   silently stop rotating — with no error surfaced anywhere. Before picking
   LOG_HOST_DIR, check whether the project directory is under any such sync
   mechanism (look for .git, a sync client config, or ask the user) and warn if the
   given path is inside one.

2. A size cap (`maxsize`) is only enforced at the moment logrotate actually runs —
   it does not watch files continuously. A daily check interval means a log can
   grow far past its cap between checks if traffic spikes. Default the sidecar's
   check loop to CHECK_INTERVAL (1h) rather than daily; only go coarser than 1h if
   the target log's realistic max growth rate per interval is well under
   MAX_SIZE_MB, and say so explicitly if you do.

3. logrotate refuses to act on a config file that is itself group/other-writable,
   or whose parent directory has insecure permissions. If the logrotate config is
   bind-mounted directly from the host, copy it to a root-owned, 644 internal path
   inside the container before every run instead of pointing logrotate at the
   mounted file directly.

4. Any log in FD_HELD_OPEN_LOGS needs `copytruncate` instead of the normal
   rename+signal rotation, because the writing process won't pick up the renamed
   file on a reopen/reload signal. Document the trade-off (a few lines can be lost
   during the copy window) in a comment next to that stanza.

DELIVERABLES:

1. docker-compose.yml — for every volume mount currently pointing at a project-relative
   log path (e.g. ./logs/) for SERVICE_NAME(S) and any READ_ONLY_CONSUMERS, repoint the
   HOST side to LOG_HOST_DIR (read-write for the writer, :ro for consumers). Leave
   container-internal mount destinations unchanged.

2. A logrotate sidecar service (add one if the project doesn't have one; reuse if it
   does):
   - Dockerfile: minimal (alpine + logrotate + docker-cli if it needs to signal other
     containers)
   - entrypoint.sh: loop that (a) self-heals the log directory's permissions to 755
     each run, (b) copies the bind-mounted logrotate.conf to a root-owned internal
     path before each invocation, (c) runs `logrotate <internal-conf> --state <path>`,
     (d) sleeps CHECK_INTERVAL, repeat. Log each run's start/end to stdout so
     `docker logs` shows rotation history.
   - logrotate.conf: one stanza per log group matching LOG_FILE_PATTERN(S), with:
     `daily`, `rotate BACKUP_COUNT`, `compress`, `delaycompress`, `missingok`,
     `notifempty`, `dateext`, `dateformat -%Y%m%d`, `maxsize MAX_SIZE_MBM`, and
     `su root root`. Use `postrotate`/reopen-signal rotation for normal logs;
     `copytruncate` for anything in FD_HELD_OPEN_LOGS.

3. A one-time migration runbook (present it, don't just run it) that:
   - Creates LOG_HOST_DIR with `sudo mkdir -p` + appropriate permissions
     (777 if multiple differently-UID'd containers write/read it; otherwise scope
     tighter)
   - COPIES (not moves) existing logs from the old location into LOG_HOST_DIR, so
     the old copy survives as a backup until the new setup is verified
   - Rebuilds the logrotate sidecar image if its Dockerfile/entrypoint changed
     (`docker compose build <logrotate-service>` — a plain `up -d` will NOT pick up
     entrypoint/Dockerfile changes, only volume/env changes)
   - Recreates only the affected services (`docker compose up -d --remove-orphans`)
   - Verifies: config test if the target has one (e.g. `nginx -t`), `docker compose
     ps` all healthy, new logs actually landing in LOG_HOST_DIR, sidecar logs show
     the new interval
   - Only after verification: deletes the old backup copies

4. Version header + changelog entries on every file touched, if the project uses that
   convention (check for a `Version:`/changelog block at the top of existing files
   before assuming).

BEFORE YOU START:
- Show the plan (files to touch, the runbook) and confirm before editing.
- Confirm before any step that requires sudo, moves data, or restarts/recreates
  live containers — these may be running production services.
- If the project's docker image uses a templating entrypoint (e.g. envsubst-rendered
  config), check whether it needs a restart vs. reload to pick up compose-level
  changes — don't assume `nginx -s reload`-equivalent hot-reload works.
```

---

## Reference implementation (example_app, for comparison)

| Parameter | Value used |
|---|---|
| LOG_HOST_DIR | `/var/log/example_app/` |
| MAX_SIZE_MB | 150 |
| BACKUP_COUNT | 1 |
| CHECK_INTERVAL | 1h (was 24h — the bug) |
| FD_HELD_OPEN_LOGS | `audit.log` (ModSecurity holds its own fd; `copytruncate` used) |
| READ_ONLY_CONSUMERS | `fail2ban`, `webui` dashboard |

Source files: `docker-compose.yml`, `logrotate/Dockerfile`, `logrotate/entrypoint.sh`,
`logrotate/logrotate.conf`, `start.sh` (log-dir pre-creation).

---
**Updated:** 2026-07-25 | v1.0.0 | Updated by: AI (Claude)
- Initial version, generalized from the example_app /var/log/example_app/ + logrotate fix. [2026-07-25 v1.0.0 AI - Claude]
