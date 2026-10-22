# Team specs

This folder declares what **team size `r`** means for each multi-agent topology, one `<dataset>.yaml` per dataset (top-level key = dataset). API-Bank and ToolHop have none: they fix their roles in code and vote over r replicas (`teamsizes/apibank_common.py`, `toolhop_common.py`).

## Layout

```
configs/teams/
└── <dataset>.yaml      # apps, bfcl, gpqa, hotpotqa, lcb, math, swe
```

| Topology | Keys |
|---|---|
| `independent` | `role` (r replicas), `recursion_limit` |
| `decentralized` | `role` (r peers), `n_rounds`, `recursion_limit` |
| `sequential` | `recursion_limit`, `tasks` (each stage's user-message template, `str.format`), `sizes`: r → the r ordered `{role, tools}` stages |
| `centralized` | `recursion_limit`, `manager_tools`, `worker_tools` (per-worker override: `tools_by_worker`), `prompt_suffixes` (optional, appended to a worker's prompt), `delegates` (each `delegate_to_<worker>` tool description, indentation included), `sizes`: r → `manager` (prompt key), `max_turns`, `workers` (r − 1) |

---

## How specs are used

- **Loaded** by `core/teams.py`: `teams.spec` returns one `TeamSpec` per `(topology, dataset, r)`, r ∈ {2, 4, 8, 10}.
- **r = 4** is the team of the `topologies/` runners (all but CrewAI); `teamsizes/` runners use their own r.
- **`recursion_limit`** bounds each agent's ReAct loop; omit it when agents make plain model calls.
- **Verbatim** — text is sent to the model as written.
