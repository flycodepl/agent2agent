# agent2agent

Cross-session agent-to-agent coordination plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Lets multiple Hermes sessions on the same host coordinate work without a human
relaying messages. Two capabilities:

- **`agent2a_steer`** — inject a prompt into another gateway session. The message
  is delivered durably (survives gateway restarts) and never interrupts a running
  turn.
- **`agent2a_lease`** — negotiate exclusive access to a shared resource (file,
  port, repo, machine) through a durable state machine stored in SQLite. The
  lease table is the source of truth; messages are only courtesy notifications.
- **`agent2a_sessions`** — discover other gateway sessions by display name,
  session id, chat id, or platform.

## How it works

### Steer (prompt injection)

`agent2a_steer` writes a one-shot `loop:<session_id>` row into the gateway's
SessionDB (`state_meta`). The gateway's built-in loop-wakeup watcher picks it up
within ~15 s and injects it as an internal user-role turn when the target
session is idle — the same durable mechanism that powers `/loop`. The target
session sees the message with a footer marking it as untrusted peer input.

If the target session is busy with a previous un-injected agent2a message, the
new message is appended to it (both delivered in one turn when the target goes
idle).

### Lease (resource negotiation)

The lease table (`~/.hermes/agent2a/leases.db`, SQLite WAL) tracks a durable
state machine per resource:

```
free  →  requested  →  held  →  free
                ↘  (denied)  ↗
```

States:

| State | Meaning |
|---|---|
| `free` | Resource is available (or was just released/denied) |
| `requested` | An agent is asking for access; the current holder (or previous user) must decide |
| `held` | A temporary holder has been granted access and is working |

Actions:

| Action | Who | Effect |
|---|---|---|
| `request` | Agent that needs the resource | Creates/updates row to `requested` with a note; notifies the holder |
| `approve` | Holder / previous user | Hands the lease over: row → `held`, holder becomes the requestor; auto-notifies |
| `deny` | Holder / previous user | Row → `free` with a reason note; auto-notifies |
| `release` | Current holder | Row → `free`; auto-notifies the previous user (the one who granted access) |
| `wait` | Requestor (or anyone) | Blocks the turn (up to `timeout` s) until the row reaches one of `until_states` (default `['held','free']`); returns the final lease row directly — no manual poll loop |
| `status` | Anyone | Returns the current row without side effects |
| `list` | Anyone | Returns all rows |
| `acquire` | Holder | Atomically takes the lease if the row is `free` (read-check-write under `BEGIN IMMEDIATE`) |
| `purge` | Anyone | Drops the row from the table |

All read-check-write operations run under `BEGIN IMMEDIATE` so concurrent
agents cannot race.

### Sessions (discovery)

`agent2a_sessions` queries the gateway routing index
(`gateway_routing(session_key, entry_json)`) and returns matching sessions
with display name, session id, platform, and chat id. Supports substring
matching (case-insensitive) and glob patterns (case-sensitive, `*`/`?`).

## Installation

Copy the plugin directory into your Hermes plugins path:

```bash
cp -r agent2agent ~/.hermes/plugins/
```

Restart the gateway (or start a new session). The tools become available in
every session automatically — no configuration needed.

Verify:

```
agent2a_sessions query='*'
```

You should see your own session and any other gateway sessions.

## Usage examples

### Example 1: Ping another agent

Session A wants to tell session B something. A calls:

```
agent2a_steer(
    target="research-agent",
    message="Heads up: the config file /etc/myapp/config.yaml was rotated at 03:00 UTC. Re-read it before your next deploy."
)
```

The message lands in B's session as a user-role turn within ~15 s, with a
footer:

```
[agent2a | from session-A (session 20260917_...)]
Heads up: the config file ...
```

B sees it, acts on it, and can reply via `agent2a_steer(target="session-A", ...)`.

### Example 2: Negotiate exclusive access to a file

Agent A is editing `/opt/deploy/config.yaml`. Agent B needs to add a line.

**B discovers the resource is in use:**

```
agent2a_lease(
    action="request",
    resource="file:/opt/deploy/config.yaml",
    note="Need to append one line (L99). ~10 s, non-destructive."
)
```

Result: `state: requested`. The holder (A) is auto-notified.

**A is busy, tells B to wait:**

```
agent2a_steer(
    target="agent-b",
    message="Got your request. Give me ~30 s — I'm finishing my last edit. You'll get an automatic notification once I approve."
)
```

**A finishes, approves with conditions:**

```
agent2a_lease(
    action="approve",
    resource="file:/opt/deploy/config.yaml",
    also_steer=True,
    note="First copy config.yaml to /opt/deploy/config.yaml.bak (BEFORE any write), then append line L99. Do not modify existing lines."
)
```

Result: `state: held`, holder = B. B is auto-notified with the conditions.

**B waits (blocking, no poll loop):**

```
agent2a_lease(
    action="wait",
    resource="file:/opt/deploy/config.yaml",
    timeout=120
)
```

Returns immediately (state is already `held`), with the full lease row
including the conditions note. B does the work.

**B finishes, releases:**

```
agent2a_lease(
    action="release",
    resource="file:/opt/deploy/config.yaml",
    note="Done: appended L99, created backup."
)
```

Result: `state: free`. A is auto-notified that the resource is free again.

### Example 3: Wait for a grant (blocking)

Agent C requests a resource and blocks until the holder decides:

```
agent2a_lease(
    action="request",
    resource="port:8443",
    note="Need to bind this port for 5 minutes."
)
agent2a_lease(
    action="wait",
    resource="port:8443",
    timeout=300
)
```

The `wait` call blocks the turn for up to 300 s. It returns the final lease
row (granted = `held` with holder = C, or resolved = `free` with a denial
note). If the timeout expires before a decision, the result carries
`timed_out: true` and the last-seen state; C can retry.

### Example 4: Discover sessions

```
agent2a_sessions(query="deploy*")
```

Returns all gateway sessions whose display name, session id, or chat name
matches the glob `deploy*` (case-sensitive). Without `*`/`?` the query is a
case-insensitive substring match.

## Security model

- **Lease table is the source of truth.** Messages are courtesy notifications
  only. If a message is lost or delayed, the state in the table is what
  matters. Always `status` before acting if unsure.
- **Untrusted peer input.** Steered messages arrive as user-role turns with a
  footer marking them as untrusted. The receiving agent is instructed (via the
  system-prompt section) to treat them as data, not instructions, and to not
  follow instructions that exceed the stated coordination scope.
- **No secret access.** The plugin does not read environment variables,
  credentials, or files outside the lease DB. It only writes to
  `~/.hermes/agent2a/leases.db` and the gateway's SessionDB.
- **Atomicity.** All read-check-write operations use `BEGIN IMMEDIATE`
  (exclusive lock). The schema migration (if an old database with a dead
  `released` state is detected) runs in a single transaction with rollback on
  failure.

## File layout

```
agent2agent/
├── __init__.py           # Plugin entry point, system-prompt section
├── core.py               # Implementation (steer, lease, sessions)
├── plugin.yaml           # Plugin manifest
├── LICENSE               # MIT
└── tests/
    ├── __init__.py
    └── test_lease_states.py   # 12 unit tests (lease state machine, migration)
```

## Running tests

```bash
cd agent2agent
python -m unittest -v tests.test_lease_states
```

Requires the Hermes venv interpreter (provides `hermes_constants` /
`hermes_cli`). Tests run against an isolated `HERMES_HOME` and never touch
the production database.

## Compatibility

- Hermes Agent v2026.4+ (uses `hermes_constants.get_hermes_home()`,
  `hermes_cli.loops`, and the gateway `state.db` routing table)
- Python 3.11+ (stdlib only — `sqlite3`, `json`, `re`, `fnmatch`, `contextlib`)
- No external dependencies

## Limitations

- **Single-host only.** Sessions must share the same gateway process (same
  `state.db`). Cross-machine coordination is out of scope — use the A2A
  protocol plugin for that.
- **One lease per resource.** The table is keyed by resource id. Two agents
  cannot hold the same resource simultaneously.
- **No TTL / auto-expiry.** A `held` lease stays `held` until the holder calls
  `release`. If a holder crashes, the row stays `held` until a human (or
  another agent) calls `release` or `purge`.
