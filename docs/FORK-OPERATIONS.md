# Fork operations

Everything in this file is specific to Clinton's fork. Upstream (`andrewyng/openworker`)
knows nothing about it. New files with fork-specific names, so the daily upstream sync
never conflicts.

## Daily upstream sync

`scripts/sync-upstream.sh`, from cron at **03:40 AEST (17:40 UTC)**, logging to
`~/.local/state/openworker-sync.log`.

It fast-forwards local `main` to `andrewyng/openworker`'s `main`, pushes that to the
`clintoncodewell/openworker` fork, then runs itself on the **Mac** over ssh so the clone
the desktop app is built from stays current too. The Mac is pull-only: the VM owns the
push, and the laptop usually has no GitHub credential anyway. A sleeping Mac logs
`unreachable (asleep?)` and the VM's own sync still counts as a success.

Remotes are resolved **by URL, not by name** — in this clone `origin` is upstream and
`fork` is ours, which is backwards from the usual convention and would silently sync the
wrong way if the script trusted the names.

It refuses a non-fast-forward. If `main` ever diverges, the log says
`FAIL: main has diverged` and nothing moves until a human decides. Feature branches are
never touched; the log's last line tells you how far behind yours is.

The Mac's clone sits on a branch called `vm-mirror`, deliberately: with `main` checked out
there, the nightly fast-forward would fight the rsync'd working tree every time.

**Auto-update is off** (`ocw.flag.autoupdate` in `surfaces/gui/src/flags.ts`). Builds come
from source, so the public update manifest could only ever offer to replace a newer local
build with an older upstream release.

## Models

Six providers are configured. Five carry a key; the sixth is the ChatGPT subscription
OAuth that was already there.

| provider | reaches | how it authenticates |
|---|---|---|
| `azure` | GPT-5.6 Sol/Terra/Luna, GPT-5.5 | key from `~/.codex/azure.key` |
| `azure-oss` | Kimi K3, DeepSeek V4 Flash | key from `~/.config/clm/azure-foundry-key` |
| `zai-coding` | GLM-5.3 / 4.6 / 4.5 | key from `~/.config/glm/key`, Anthropic-shaped endpoint |
| `gemini` | Gemini 3.6 Flash, 3.1 Pro, 2.5 | **free-tier AI Studio key only** — see below |
| `xai` | Grok 4.6 / 4.5 / 4.3 | none — through the VM's LiteLLM proxy |
| `claude-code` | Claude Opus 5 / Sonnet 5 / Haiku 4.5 | none — your Claude Code subscription |

`scripts/configure-providers.py` collects those keys into OpenWorker's own SecretStore and
verifies each with one read-only call. Re-run it after any key rotates:

```shell
.venv/bin/python scripts/configure-providers.py            # collect, apply, verify
.venv/bin/python scripts/configure-providers.py --verify-only
```

### Claude on the subscription, not an API key

`coworker/providers/claude_code_provider.py` shells out to `claude -p`, which authenticates
with the OAuth credentials Claude Code already holds. No `sk-ant-` key to store, rotate or
leak; nothing is billed per token.

**It is text-only, on purpose.** The CLI answers in prose, never `tool_use` blocks, so the
provider declares `tools=False` and refuses a tools list rather than returning a turn that
looks like the model declined to use them. It is a first-class **council member**; it
cannot be the session model. For that reason its models are deliberately **not** in the
curated matrix (which feeds the session-model picker) — a test enforces that.

**The catch is overhead, and it is large.** Every call re-sends Claude Code's own harness:

| Invocation | Prompt tokens | At list price |
|---|---:|---:|
| plain `claude -p "reply ok"` | ~36,000 | $0.37 |
| this provider's stripped invocation | ~12,000 | $0.12 |
| the same question via the API | ~93 | ~$0.0001 |

So this is not the cheap option, it is the **no-API-key** option. Those tokens come out of
your Claude Code rate limits — the same pool your coding sessions use — which is the budget
to watch. The council's per-model spend table shows the token count against `$0.0000`,
which is the honest picture: no dollars, real allowance.

Two isolation details, both learned the hard way:

- It runs in an **empty scratch directory**, because Claude Code discovers `CLAUDE.md`,
  skills and settings from its working directory. Without that, one panel member argues
  from whichever repo the app is open on while the others do not.
- **`--bare` cannot be used.** It would disable global `CLAUDE.md` and auto-memory, but it
  also skips keychain reads, which is where the subscription credentials live — the call
  fails with "Not logged in". A small amount of account-level context (the signed-in email)
  still reaches the model; project and memory context does not.

Keyless does not mean available here: `provider_configured` runs a `which claude`, so an
uninstalled CLI is reported as unconfigured instead of putting a dead member on the panel.

### Gemini uses the free tier, never the Cloud project

Set 2026-08-16. `GEMINI_API_KEY` and `GOOGLE_API_KEY` are the Google SDK's conventional
names, so on this box they hold the key belonging to the **`aw-gemini-api-central` GCP
project** — billed work, shared with the whole OpenClaw stack. OpenWorker must not spend
that account.

So the Gemini provider ignores both names. It reads, in order:

1. the key stored in Settings ▸ Models, or
2. `OPENWORKER_GEMINI_API_KEY` — a name nothing else on this box sets.

No key in either place and the provider refuses to build, loudly. It never falls back:
a silent fallback to the billed key is precisely the failure this prevents, and "remove
key" in Settings could not have turned it off.

**To turn Gemini back on**, get a key from
[aistudio.google.com](https://aistudio.google.com/apikey) under the personal Gmail
account, then either paste it into Settings ▸ Models, or:

```shell
install -m 600 /dev/null ~/.config/coworker/gemini-aistudio-key
printf %s 'AIza…' > ~/.config/coworker/gemini-aistudio-key
.venv/bin/python scripts/configure-providers.py
```

`configure-providers.py` reads only that dedicated file — it will not harvest the shared
env key, by design.

**OpenClaw was switched off Gemini at the same time.** Its `memorySearch` ran on
`gemini-embedding-001`; it now runs `local` embeddings via the `@openclaw/llama-cpp-provider`
plugin (embeddinggemma-300m, on this box, no key, no billing). The `google` plugin is
disabled and both Gemini keys are gone from `gateway.systemd.env`. The memory index was
rebuilt for the new model — a change of embedding model **requires** a reindex, or vector
search silently pauses:

```shell
openclaw memory index --agent main && openclaw memory status --deep
```

**Endpoints are per-tenant and are NOT prefilled.** Both Foundry providers ship with an
empty endpoint on purpose: a Foundry resource has its own hostname, key, and deployment
names, so baking one tenant's hostname in as everyone's default would send a stranger's
key to our resource. The build fails loudly with "No … endpoint configured" instead of
quietly falling back to `api.openai.com`.

### Getting the same models onto the Mac

The Mac has none of the key files, so the keys travel as a bundle. It is plaintext key
material — ship it over ssh and delete both copies.

```shell
# on the VM
.venv/bin/python scripts/configure-providers.py --out /tmp/bundle.json --proxy-host 100.65.245.83
scp /tmp/bundle.json mac:/tmp/ && rm /tmp/bundle.json
# on the Mac
.venv/bin/python scripts/configure-providers.py --apply /tmp/bundle.json && rm /tmp/bundle.json
```

`--proxy-host` matters: two providers do not work off the VM without it (below).

## The two proxies

Both run as user systemd units on the VM and bind **only** the tailnet address
`100.65.245.83` — never `0.0.0.0`. Only Clinton's own devices can reach them.

**`openclaw-grok-proxy` + `grok-tailnet-forward` (port 4144).** Grok authenticates with an
xAI OAuth bearer. The proxy fetches a fresh one on every restart, so nothing downstream
ever holds a token. `grok-tailnet-forward.service` is a socat hop that publishes the
loopback-only proxy on the tailnet for the Mac.

> **A new Grok model needs a proxy entry first.** The proxy is an allowlist, not a
> pass-through: a model absent from `~/.grokcc/litellm.config.yaml` returns
> `400 Invalid model name` even though xAI serves it. Add a `model_list` entry and restart
> the unit. (Done for `grok-4.6` on 2026-08-16; `.bak-pre-grok46` is the previous config.)
>
> **If Grok 403s with `unauthenticated:bad-credentials`, restart the proxy:**
> `systemctl --user restart openclaw-grok-proxy`.
> The proxy captures its bearer at start and holds it in memory. The unit already cycles
> every 4h against a 6h token, but any *other* grok consumer (the `grokcc` CLI) that
> refreshes the token rotates it and invalidates the copy the running proxy is holding —
> so the staleness is triggered by use elsewhere, not by elapsed time. A council with a
> dead Grok still reports; that member is listed under `failures`.

**`openworker-foundry-proxy` (port 8802).** The `foundry-codex-dev` Foundry resource has a
VNet/firewall rule allowing only the VM's IP, so the Mac gets a 403 with a perfectly valid
key. `scripts/foundry-tailnet-proxy.py` forwards to it over TLS and rewrites `Host` (Azure's
front door rejects any other hostname). It holds no credentials — the caller's
`Authorization` header passes straight through — and it refuses to bind anything outside
`100.64.0.0/10` or loopback.

The alternative was adding each machine's public IP to the Foundry firewall. Home IPs
rotate, and a public IP is a much broader grant than a tailnet identity.

```shell
systemctl --user status openworker-foundry-proxy grok-tailnet-forward
```

## The council

`coworker/council/`. One tool call puts a question to every configured model, each arguing
a **different assigned lens**, has them rebut each other over a shared scratchpad, then a
chair model writes the finding. Registered for every agent, so any chat can ask: *"convene
the council on X"*, *"what do all the models think"*.

Everything below is editable in **Settings ▸ Council**.

### Two modes

**Analysis** — the general panel. **Decision** — for a real choice with stakes. Decision
mode makes members name the options nobody listed, state what would have to be TRUE, run a
**pre-mortem** ("it is a year later and this went badly — write how"), and rate
reversibility. The chair then has to commit: a recommendation, the two or three factors
that actually decide it, the one assumption it rests on, the most likely failure, and a
dated signal to change course.

Ask for it by name: *"convene a decision council on whether I should take the job"*.

### Why it is built this way

The multi-agent-debate literature is blunt about how these systems fail, and each design
choice here answers one of those failures:

| Failure mode | What we do |
|---|---|
| Identical agents ≈ one agent | A different lens per member, plus five different vendors |
| Sycophancy, disagreement collapse | An explicit hold-your-position rule every debate round |
| A weak judge wastes a strong panel | The chair model is separately configurable |
| Arguing when everyone agrees | The debate round is skipped when round 1 already concurs |
| Information fragmentation | A shared scratchpad every member reads and writes |

The heterogeneous panel is the load-bearing part: homogeneous panels provably cannot beat
a majority vote, because they converge. Five vendors with different base models and
different post-training is a much stronger defence than five personas on one model.

### Scoped source data

Point the panel at your own material and it argues from that instead of the open web.
Every source resolves once, and the same brief goes to every member.

| Kind | Target | Notes |
|---|---|---|
| `folder` | a directory | `{"glob": "**/*.md"}` under Options |
| `file` | one file | |
| `url` | a web page | stripped to text |
| `search` | a query | uses the app's search provider |
| `http` | any GET API | credentials via `{"headers_profile": "api:mine"}` |
| `mcp` | `server:tool` | the door for knowledge bases and databases |

There is no `database` kind on purpose: Postgres, SQLite and Supabase all ship MCP
servers, so `mcp` already covers them without a second credential path.

**Credentials never go in `council.json`.** For `http`, name a SecretStore profile and its
values are sent as headers. Test any source before saving — the Test button reports what
actually came back, so a wrong glob is a visible error instead of a quietly thinner brief.

**Everything you attach is sent to every configured vendor.** Scope sources to what the
question needs.

### The scratchpad

Each member ends its answer with one `NOTE:` line worth sharing. Those accumulate into a
blackboard every member reads in the next round, and the chair reads at the end. Each run
writes three files under `~/.config/coworker/council/<date>-<slug>/`:

- `finding.md` — the chair's answer, panel, dissent
- `scratchpad.md` — the shared notes, by round and author
- `transcript.md` — every member's full text

Browse them in Settings ▸ Council ▸ History. They are plain markdown on purpose: the
reasoning behind a decision is still readable in a year, when the session is long gone.

### What it costs — measured, not guessed

Every run reports its own spend, so you never have to wonder. `finding.md` and the tool
result both carry a line like `11 model calls · 32,941 tokens, 3,075 of it hidden
reasoning · about $0.075`, plus a per-model breakdown showing which member drove the bill.

Two real runs on the five-provider panel:

| Run | Calls | Tokens | Cost |
|---|---:|---:|---:|
| 2 rounds, web search, no sources | 11 | 33k | **$0.08** |
| 3 rounds, decision mode, 10k-char source | 16 | 115k | **$0.24** |

So a council is **cents, not dollars** — you would need roughly 4,000 of them to spend
A$1,500. It cannot "argue for days": the loop is a fixed number of rounds, not a
conversation that continues until agreement.

Five things bound it structurally, before any budget is involved:

1. **Round count is fixed** at 1-3. There is no "keep going until they agree" path.
2. **8 members max**, so the worst case is 25 calls.
3. **Answers are capped** by the prompts at 250-350 words each.
4. **Source material is capped** at 120k characters total, and it is the multiplier that
   matters — every source is re-sent to every member on every round.
5. **The debate is skipped** when round one already agrees.

On top of that, `max_tokens_per_run` (default 500,000, about 4x the most expensive real run
above) bounds how many **rounds** get added. Be clear on what it is not: it is checked
*between* rounds, so the opening round is already spent when it first applies and the chair
always runs afterwards — a run can therefore finish above the figure. What it prevents is a
council that keeps adding rounds past it. Making it a true hard cap would need per-provider
pre-flight token estimation, which is a lot of machinery for a guard that exists to catch
one pathological case. The result carries `stopped_on_budget: true`.

**Tokens are measured; dollars are an estimate.** The token counts come from each vendor's
own response. The dollar figure comes from a hand-maintained price table in
`coworker/council/usage.py`, checked 2026-08-16. Re-check before making a decision on it.

Watch for **hidden reasoning tokens** — they are billed as output and are frequently
larger than the visible answer. One Gemini call answering "Ready?" in one word billed 151
thinking tokens against 9 of prompt. The spend line calls them out separately.

### Speed

Measured across the five configured providers: **about 60-110 seconds** for two rounds.
Kimi K3 is consistently the slowest member at 15-40s; GLM the fastest at 3-8s. Members run
concurrently, so the wall clock is the slowest member per round, not the sum.

### Turning the cost down

In rough order of effect:

- **Drop to 1 round** (Settings ▸ Council ▸ Rounds). Halves it; you lose the rebuttal.
- **Trim sources.** They are the biggest multiplier — a 10k-char folder cost 3x here.
- **Turn off the web search** if the question is not about current facts.
- **Shrink the panel** by pointing `panel` at three models instead of all six.
- **Leave the agreement-skip on.** It is free money when the panel concurs.

Two things worth knowing:

- **It is `risk_level="medium"`, not `"low"`.** The engine runs low-risk tools
  concurrently, so as a low-risk tool a single turn asking for five councils would put
  ~125 paid completions in flight at once. Medium keeps it auto-approved but strictly
  serial.
- **The consensus comes back labelled as untrusted data**, because it is assembled from
  model output, your source material and web snippets, and it lands in an agent that has
  shell and write tools.
