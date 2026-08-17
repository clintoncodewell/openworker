// Settings ▸ Council — the panel's brain, in four panes.
//
// The council is the one feature where the prompt IS the product: what the members are
// told to argue decides whether five models produce insight or five paraphrases of the
// same answer. So the prompts are first-class editable text, not buried constants, and
// every one shows a Reset that restores the shipped wording.
//
// Panes: Panel (who argues, in which role) · Prompts · Sources (scoped input data) ·
// History (past runs on disk). Each saves independently — the server merges partials.
import { useEffect, useRef, useState } from "react";
import {
  getCouncilConfig,
  getCouncilRun,
  getCouncilRuns,
  setCouncilConfig,
  testCouncilSource,
  type CouncilConfig,
  type CouncilSource,
} from "../api";
import { Icon } from "./Icon";
import { Markdown } from "./Markdown";

const CARD = "rounded-xl2 border border-line bg-panel";
const INPUT =
  "w-full min-w-0 px-3 py-2 rounded-lg border border-line bg-paper text-[13px] text-ink outline-none focus:border-accent";
const BTN = "text-[12.5px] px-3 py-2 rounded-lg border border-line bg-paper hover:border-lineStrong shrink-0";
const BTN_ACCENT = "text-[12.5px] px-3 py-2 rounded-lg bg-accent text-white shrink-0 disabled:opacity-40";
const HELP = "text-[12px] text-muted mt-1.5 leading-relaxed";

type Pane = "panel" | "prompts" | "sources" | "history";
const PANES: { key: Pane; label: string }[] = [
  { key: "panel", label: "Panel" },
  { key: "prompts", label: "Prompts" },
  { key: "sources", label: "Sources" },
  { key: "history", label: "History" },
];

const PHASES: { key: string; label: string; help: string }[] = [
  { key: "round1", label: "Opening round", help: "Each member answers alone. {role_name} and {role_brief} are filled in per member." },
  { key: "debate", label: "Debate round", help: "Members see each other's answers. {me} is the member's own model id; {anti_conformity} is the hold-your-position rule." },
  { key: "chair", label: "Chair", help: "Reads the whole transcript and writes the finding." },
];

export function CouncilTab() {
  const [cfg, setCfg] = useState<CouncilConfig | null>(null);
  const [pane, setPane] = useState<Pane>("panel");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getCouncilConfig().then(setCfg).catch(() => {});
  }, []);

  const save = async (patch: Partial<CouncilConfig>) => {
    const next = await setCouncilConfig(patch);
    setCfg(next);
    setSaved(true);
    window.setTimeout(() => setSaved(false), 1800);
  };

  if (!cfg) return <div className="text-[13px] text-muted">Loading…</div>;

  return (
    <div>
      <div className="flex items-center gap-1 mb-4" role="tablist">
        {PANES.map((p) => (
          <button
            key={p.key}
            role="tab"
            aria-selected={pane === p.key}
            className={
              "text-[12.5px] px-3 py-1.5 rounded-lg " +
              (pane === p.key ? "bg-panel text-accent font-medium" : "text-muted hover:text-ink")
            }
            onClick={() => setPane(p.key)}
          >
            {p.label}
          </button>
        ))}
        <span
          className={
            "ml-auto text-[12px] text-muted transition-opacity " + (saved ? "opacity-100" : "opacity-0")
          }
          aria-live="polite"
        >
          Saved
        </span>
      </div>

      {pane === "panel" ? (
        <PanelPane cfg={cfg} save={save} />
      ) : pane === "prompts" ? (
        <PromptsPane cfg={cfg} save={save} />
      ) : pane === "sources" ? (
        <SourcesPane cfg={cfg} save={save} />
      ) : (
        <HistoryPane />
      )}
    </div>
  );
}

// -- how hard to think, and how much to write ------------------------------------
// One control each, with the cost stated. Rounds and panel size move together because
// picking them separately lets you choose a pair that makes no sense — three rounds of two
// members — and because "how much compute do I want on this" is the question actually
// being asked. Custom puts the raw fields back.

function OptionCards<T extends string>({
  label,
  help,
  value,
  options,
  onPick,
  testid,
}: {
  label: string;
  help?: string;
  value: T;
  options: { key: T; label: string; blurb: string }[];
  onPick: (key: T) => void;
  testid: string;
}) {
  return (
    <div className="block">
      <span className="text-[12.5px] font-medium text-ink">{label}</span>
      <div className="mt-2 grid gap-2" data-testid={testid}>
        {options.map((o) => {
          const on = o.key === value;
          return (
            <button
              key={o.key}
              type="button"
              role="radio"
              aria-checked={on}
              onClick={() => onPick(o.key)}
              className={
                "text-left rounded-xl border px-3 py-2.5 transition-colors " +
                (on
                  ? "border-accent bg-accent/5"
                  : "border-line hover:border-muted hover:bg-paper")
              }
            >
              <div className="flex items-center gap-2">
                <span
                  className={
                    "w-3.5 h-3.5 rounded-full border grid place-items-center shrink-0 " +
                    (on ? "border-accent" : "border-line")
                  }
                  aria-hidden
                >
                  {on && <span className="w-1.5 h-1.5 rounded-full bg-accent" />}
                </span>
                <span className="text-[12.5px] font-medium text-ink">{o.label}</span>
              </div>
              <div className="text-[12px] text-muted mt-1 leading-relaxed pl-[22px]">{o.blurb}</div>
            </button>
          );
        })}
      </div>
      {help && <p className={HELP}>{help}</p>}
    </div>
  );
}

function DepthPicker({
  cfg,
  save,
}: {
  cfg: CouncilConfig;
  save: (p: Partial<CouncilConfig>) => Promise<void>;
}) {
  const depths = cfg.depths || {};
  return (
    <OptionCards
      label="How hard should it think?"
      testid="depth-picker"
      value={cfg.depth}
      onPick={(depth) => save({ depth })}
      options={[
        ...Object.entries(depths).map(([key, d]) => ({
          key,
          label: d.label,
          blurb: d.blurb,
        })),
        {
          key: "custom",
          label: "Custom",
          blurb: "Set the rounds and the web search yourself.",
        },
      ]}
    />
  );
}

function DetailPicker({
  cfg,
  save,
}: {
  cfg: CouncilConfig;
  save: (p: Partial<CouncilConfig>) => Promise<void>;
}) {
  const details = cfg.details || {};
  return (
    <OptionCards
      label="How much of the finding do you want?"
      help="Every length opens with the answer in two sentences, so you can stop reading there."
      testid="detail-picker"
      value={cfg.detail}
      onPick={(detail) => save({ detail })}
      options={Object.entries(details).map(([key, d]) => ({
        key,
        label: d.label,
        blurb: d.blurb,
      }))}
    />
  );
}

// -- panel -----------------------------------------------------------------------

function PanelPane({
  cfg,
  save,
}: {
  cfg: CouncilConfig;
  save: (p: Partial<CouncilConfig>) => Promise<void>;
}) {
  return (
    <div className="space-y-4">
      <div className={CARD + " p-4"}>
        <div className="text-[13px] font-medium mb-1">Who is on the panel</div>
        <p className={HELP}>
          One model per configured provider, each assigned a different lens. The lenses are the
          point: a panel of identically-briefed models barely beats asking one of them.
        </p>
        <ul className="mt-3 space-y-1.5" data-testid="resolved-panel">
          {cfg.resolved_panel.map((m) => (
            <li key={m.model} className="flex items-center gap-2 text-[13px]">
              <span className="tabular-nums text-muted w-[9.5rem] shrink-0 truncate">{m.role}</span>
              <span className="text-ink truncate">{m.model}</span>
            </li>
          ))}
          {cfg.resolved_panel.length === 0 && (
            <li className="text-[13px] text-muted">
              No providers configured yet — add a model key under Settings ▸ Models.
            </li>
          )}
        </ul>
        <div className="mt-3 text-[12px] text-muted">
          Chair: <span className="text-ink">{cfg.resolved_chair || "the session's model"}</span>
        </div>
      </div>

      <div className={CARD + " p-4 space-y-4"}>
        <label className="block">
          <span className="text-[12.5px] font-medium text-ink">Chair model</span>
          <input
            className={INPUT + " mt-1.5"}
            aria-label="Chair model"
            defaultValue={cfg.chair_model}
            placeholder="leave blank to use the session's model"
            onBlur={(e) => e.target.value !== cfg.chair_model && save({ chair_model: e.target.value.trim() })}
          />
          <p className={HELP}>
            Worth setting to your strongest model. A weak chair cannot adjudicate between strong
            arguments, and wastes the whole panel.
          </p>
        </label>

        <label className="block">
          <span className="text-[12.5px] font-medium text-ink">Default mode</span>
          <select
            aria-label="Default mode"
            className={INPUT + " mt-1.5"}
            value={cfg.preset}
            onChange={(e) => save({ preset: e.target.value })}
          >
            <option value="analysis">Analysis — what do the models think</option>
            <option value="decision">Decision — a real choice with stakes</option>
          </select>
          <p className={HELP}>
            Decision mode adds options, assumptions, a pre-mortem and reversibility, and makes the
            chair commit to a recommendation.
          </p>
        </label>

        <DepthPicker cfg={cfg} save={save} />
        <DetailPicker cfg={cfg} save={save} />

        {cfg.depth === "custom" && (
          <>
            <label className="block">
              <span className="text-[12.5px] font-medium text-ink">Rounds</span>
              <select
                aria-label="Rounds"
                className={INPUT + " mt-1.5"}
                value={String(cfg.rounds)}
                onChange={(e) => save({ rounds: Number(e.target.value) })}
              >
                <option value="1">1 — independent answers only</option>
                <option value="2">2 — one rebuttal round</option>
                <option value="3">3 — two rebuttal rounds</option>
              </select>
            </label>

            <Toggle
              label="Search the web before the panel answers"
              checked={cfg.research}
              onChange={(v) => save({ research: v })}
            />
          </>
        )}

        <Toggle
          label="Skip the debate when the opening round already agrees"
          help="Saves a full round of paid calls when there is nothing to argue about."
          checked={cfg.skip_debate_on_agreement}
          onChange={(v) => save({ skip_debate_on_agreement: v })}
        />

        <label className="block">
          <span className="text-[12.5px] font-medium text-ink">Stop adding rounds past</span>
          <input
            className={INPUT + " mt-1.5 tabular-nums"}
            aria-label="Stop adding rounds past"
            type="number"
            min={0}
            step={50000}
            defaultValue={cfg.max_tokens_per_run}
            onBlur={(e) =>
              Number(e.target.value) !== cfg.max_tokens_per_run &&
              save({ max_tokens_per_run: Math.max(0, Number(e.target.value) || 0) })
            }
          />
          <p className={HELP}>
            Checked between rounds, so it bounds how many rounds get added — it is not a hard
            cap. The opening round is already spent when it first applies, and the chair still
            runs afterwards so you get the finding you paid for, which means a run can finish
            above this figure. 0 turns the guard off. A typical two-round council uses about
            35,000 tokens.
          </p>
        </label>
      </div>

      <RolesCard cfg={cfg} save={save} />
    </div>
  );
}

function RolesCard({
  cfg,
  save,
}: {
  cfg: CouncilConfig;
  save: (p: Partial<CouncilConfig>) => Promise<void>;
}) {
  const [roles, setRoles] = useState(cfg.roles);
  const edit = (i: number, key: "name" | "brief", value: string) =>
    setRoles(roles.map((r, n) => (n === i ? { ...r, [key]: value } : r)));

  return (
    <div className={CARD + " p-4"}>
      <div className="flex items-center gap-2">
        <div className="text-[13px] font-medium">Lenses</div>
        <button
          className={BTN + " ml-auto"}
          onClick={() => {
            setRoles(cfg.default_roles);
            save({ roles: cfg.default_roles });
          }}
        >
          Reset
        </button>
      </div>
      <p className={HELP}>Assigned to members in order, wrapping round if the panel is longer.</p>
      <div className="mt-3 space-y-2">
        {roles.map((r, i) => (
          <div key={i} className="flex gap-2">
            <input
              className={INPUT + " w-[9.5rem] shrink-0"}
              value={r.name}
              aria-label={`Lens ${i + 1} name`}
              onChange={(e) => edit(i, "name", e.target.value)}
            />
            <input
              className={INPUT}
              value={r.brief}
              aria-label={`Lens ${i + 1} brief`}
              onChange={(e) => edit(i, "brief", e.target.value)}
            />
            <button
              className={BTN}
              aria-label={`Remove lens ${i + 1}`}
              onClick={() => setRoles(roles.filter((_, n) => n !== i))}
            >
              <Icon name="trash" size={14} />
            </button>
          </div>
        ))}
      </div>
      <div className="mt-3 flex gap-2">
        <button className={BTN} onClick={() => setRoles([...roles, { name: "", brief: "" }])}>
          <Icon name="plus" size={14} /> Add lens
        </button>
        <button className={BTN_ACCENT} onClick={() => save({ roles: roles.filter((r) => r.name.trim()) })}>
          Save lenses
        </button>
      </div>
    </div>
  );
}

// -- prompts ---------------------------------------------------------------------

function PromptsPane({
  cfg,
  save,
}: {
  cfg: CouncilConfig;
  save: (p: Partial<CouncilConfig>) => Promise<void>;
}) {
  const [preset, setPreset] = useState(cfg.preset);
  const overrides = cfg.prompts[preset] || {};

  const write = (phase: string, text: string) =>
    save({
      prompts: {
        ...cfg.prompts,
        [preset]: { ...(cfg.prompts[preset] || {}), [phase]: text },
      },
    });

  return (
    <div className="space-y-4">
      <div className={CARD + " p-4"}>
        <label className="block">
          <span className="text-[12.5px] font-medium text-ink">Editing prompts for</span>
          <select
            aria-label="Editing prompts for"
            className={INPUT + " mt-1.5"}
            value={preset}
            onChange={(e) => setPreset(e.target.value)}
          >
            <option value="analysis">Analysis mode</option>
            <option value="decision">Decision mode</option>
          </select>
          <p className={HELP}>
            Each mode has its own three prompts. Leave one untouched and it keeps tracking the
            shipped wording, so an improvement here still reaches you.
          </p>
        </label>
      </div>

      {PHASES.map((p) => (
        <PromptEditor
          key={`${preset}:${p.key}`}
          label={p.label}
          help={p.help}
          shipped={cfg.defaults[preset]?.[p.key] || ""}
          value={overrides[p.key] || ""}
          onSave={(text) => write(p.key, text)}
        />
      ))}
    </div>
  );
}

function PromptEditor({
  label,
  help,
  shipped,
  value,
  onSave,
}: {
  label: string;
  help: string;
  shipped: string;
  value: string;
  onSave: (text: string) => void;
}) {
  const [text, setText] = useState(value || shipped);
  const edited = Boolean(value);
  // Keying the component on preset+phase remounts it on a mode switch, so this only has
  // to follow a save of the SAME field.
  const last = useRef(value);
  useEffect(() => {
    if (last.current !== value) {
      last.current = value;
      setText(value || shipped);
    }
  }, [value, shipped]);

  return (
    <div className={CARD + " p-4"}>
      <div className="flex items-center gap-2">
        <div className="text-[13px] font-medium">{label}</div>
        {edited && (
          <span className="text-[11.5px] px-1.5 py-0.5 rounded bg-paper border border-line text-muted">
            edited
          </span>
        )}
        <button
          className={BTN + " ml-auto"}
          disabled={!edited && text === shipped}
          onClick={() => {
            setText(shipped);
            onSave("");
          }}
        >
          Reset
        </button>
        <button className={BTN_ACCENT} disabled={text === (value || shipped)} onClick={() => onSave(text)}>
          Save
        </button>
      </div>
      <p className={HELP}>{help}</p>
      <textarea
        className={INPUT + " mt-2.5 font-mono text-[12px] leading-relaxed"}
        rows={12}
        aria-label={label}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
    </div>
  );
}

// -- sources ---------------------------------------------------------------------

const KIND_HELP: Record<string, string> = {
  folder:
    "A directory of text files. Set a glob under Options, e.g. {\"glob\": \"**/*.md\"}. Dotfiles are never read — that is where credentials live, and everything here goes to every vendor.",
  file: "One file.",
  url: "A web page, stripped to text.",
  search: "A web search. The target is the query.",
  http:
    "A GET against any API. Put credentials in a SecretStore profile and name it under Options as headers_profile — never paste a key here. Local and private addresses need {\"allow_local\": true}; cloud-metadata addresses are always refused.",
  mcp: "One MCP tool call, as server:tool. This is the door for knowledge bases and databases. Arguments go under Options.",
};

function SourcesPane({
  cfg,
  save,
}: {
  cfg: CouncilConfig;
  save: (p: Partial<CouncilConfig>) => Promise<void>;
}) {
  const [sources, setSources] = useState<CouncilSource[]>(cfg.sources);
  const [tested, setTested] = useState<Record<number, string>>({});
  // Which rows currently hold unparseable Options JSON. Saving or testing one of those
  // would silently use the last value that DID parse, which is how a typo becomes a
  // source that quietly behaves differently from what is on screen.
  const [badJson, setBadJson] = useState<Record<number, boolean>>({});
  const anyBadJson = Object.values(badJson).some(Boolean);

  const edit = (i: number, patch: Partial<CouncilSource>) =>
    setSources(sources.map((s, n) => (n === i ? { ...s, ...patch } : s)));

  const test = async (i: number) => {
    // Functional updates throughout: testing two sources at once with `{...tested}` reads
    // a stale snapshot and drops one result. And a rejected fetch would otherwise leave
    // the row on "Testing…" forever with an unhandled rejection behind it.
    setTested((t) => ({ ...t, [i]: "Testing…" }));
    try {
      const r = await testCouncilSource(sources[i]);
      setTested((t) => ({
        ...t,
        [i]: r.ok
          ? `OK — ${r.chars} characters${r.truncated ? " (truncated)" : ""}`
          : `Failed: ${r.error}`,
      }));
    } catch (e) {
      setTested((t) => ({ ...t, [i]: `Failed: ${e instanceof Error ? e.message : String(e)}` }));
    }
  };

  return (
    <div className="space-y-4">
      <div className={CARD + " p-4"}>
        <div className="text-[13px] font-medium mb-1">Scoped input data</div>
        <p className={HELP}>
          Every source is resolved once and shown to every member, so the panel argues from your
          material rather than whatever it can find. Everything here is sent to every configured
          vendor — scope it to what the question needs.
        </p>
      </div>

      {sources.map((s, i) => (
        <div key={i} className={CARD + " p-4 space-y-2.5"}>
          <div className="flex gap-2">
            <select
              className={INPUT + " w-[7.5rem] shrink-0"}
              value={s.kind}
              aria-label={`Source ${i + 1} kind`}
              onChange={(e) => edit(i, { kind: e.target.value })}
            >
              {cfg.source_kinds.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
            <input
              className={INPUT}
              value={s.target}
              placeholder="path, URL, query, or server:tool"
              aria-label={`Source ${i + 1} target`}
              onChange={(e) => edit(i, { target: e.target.value })}
            />
            <button className={BTN} aria-label={`Remove source ${i + 1}`} onClick={() => setSources(sources.filter((_, n) => n !== i))}>
              <Icon name="trash" size={14} />
            </button>
          </div>
          <div className="flex gap-2">
            <input
              className={INPUT}
              value={s.label}
              placeholder="label shown to the panel"
              aria-label={`Source ${i + 1} label`}
              onChange={(e) => edit(i, { label: e.target.value })}
            />
            <OptionsField
              value={s.options || {}}
              label={`Source ${i + 1} options`}
              onChange={(options) => edit(i, { options })}
              onValidity={(ok) => setBadJson({ ...badJson, [i]: !ok })}
            />
          </div>
          <p className={HELP}>{KIND_HELP[s.kind] || ""}</p>
          <div className="flex items-center gap-2">
            <Toggle label="Enabled" checked={s.enabled} onChange={(v) => edit(i, { enabled: v })} />
            <button className={BTN + " ml-auto"} disabled={badJson[i]} onClick={() => test(i)}>
              Test
            </button>
          </div>
          {tested[i] && (
            <div className="text-[12px] text-muted" role="status">
              {tested[i]}
            </div>
          )}
        </div>
      ))}

      <div className="flex gap-2">
        <button
          className={BTN}
          onClick={() =>
            setSources([...sources, { kind: "folder", target: "", label: "", options: {}, enabled: true }])
          }
        >
          <Icon name="plus" size={14} /> Add source
        </button>
        <button
          className={BTN_ACCENT}
          disabled={anyBadJson}
          onClick={() => save({ sources: sources.filter((s) => s.target.trim()) })}
        >
          Save sources
        </button>
        {anyBadJson && (
          <span className="text-[12px] text-muted self-center" role="status">
            Fix the highlighted Options JSON first.
          </span>
        )}
      </div>
    </div>
  );
}

/** A JSON object field that keeps the user's RAW text while they type.
 *
 * Deriving the input's value from the parsed object means every keystroke that makes the
 * JSON temporarily invalid — the first `{`, a half-typed key — is discarded and React
 * puts the old text back, so the field can only really be edited by pasting. Hold the
 * draft, parse alongside it, and report validity up so Save and Test can wait. */
function OptionsField({
  value,
  label,
  onChange,
  onValidity,
}: {
  value: Record<string, any>;
  label: string;
  onChange: (v: Record<string, any>) => void;
  onValidity: (ok: boolean) => void;
}) {
  const [draft, setDraft] = useState(() => JSON.stringify(value));
  const [bad, setBad] = useState(false);

  const type = (text: string) => {
    setDraft(text);
    try {
      const parsed = JSON.parse(text.trim() || "{}");
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("not an object");
      }
      setBad(false);
      onValidity(true);
      onChange(parsed);
    } catch {
      setBad(true);
      onValidity(false);
    }
  };

  return (
    <input
      className={INPUT + " font-mono text-[12px] " + (bad ? "border-danger" : "")}
      value={draft}
      aria-label={label}
      aria-invalid={bad}
      onChange={(e) => type(e.target.value)}
    />
  );
}

// -- history ----------------------------------------------------------------------

function HistoryPane() {
  const [runs, setRuns] = useState<{ id: string; updated_at: number }[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  // Keyed by run id, not a bare files map: with one shared map, opening run B renders B's
  // header over A's still-loaded markdown until the fetch lands — and if the fetch fails,
  // A's transcript stays on screen labelled as B.
  const [files, setFiles] = useState<Record<string, Record<string, string>>>({});
  const [error, setError] = useState<Record<string, string>>({});
  // The run list has three states, not two. Swallowing a fetch failure and falling through
  // to the empty-state copy tells the user they have no councils when in fact the request
  // failed — the one message guaranteed to send them looking in the wrong place.
  const [listState, setListState] = useState<"loading" | "ready" | "failed">("loading");

  useEffect(() => {
    getCouncilRuns()
      .then((r) => {
        setRuns(r);
        setListState("ready");
      })
      .catch(() => setListState("failed"));
  }, []);

  const show = async (id: string) => {
    if (open === id) return setOpen(null);
    setOpen(id);
    if (files[id]) return; // already fetched
    // Clear any previous failure for this run, or a retry that succeeds still renders the
    // stale error beside the freshly loaded files.
    setError((e) => {
      const { [id]: _dropped, ...rest } = e;
      return rest;
    });
    try {
      const r = await getCouncilRun(id);
      if (r.files) setFiles((f) => ({ ...f, [id]: r.files as Record<string, string> }));
      else setError((e) => ({ ...e, [id]: r.error || "could not read this run" }));
    } catch (e) {
      setError((prev) => ({ ...prev, [id]: e instanceof Error ? e.message : String(e) }));
    }
  };

  if (listState === "loading") {
    return <div className={CARD + " p-4 text-[13px] text-muted"}>Loading…</div>;
  }

  if (listState === "failed") {
    return (
      <div className={CARD + " p-4 text-[13px] text-muted"} role="status">
        Could not load past councils. The server may not be running — this is not the same
        as having none.
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className={CARD + " p-4 text-[13px] text-muted"}>
        No councils yet. Ask for one in any chat — "convene the council on…" — and the finding,
        the transcript and the panel's scratchpad land here.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {runs.map((r) => (
        <div key={r.id} className={CARD + " p-4"}>
          <button className="w-full text-left flex items-center gap-2" onClick={() => show(r.id)}>
            <Icon name={open === r.id ? "chevronDown" : "chevronRight"} size={14} />
            <span className="text-[13px] truncate">{r.id}</span>
          </button>
          {open === r.id && (
            <div className="mt-3 space-y-3">
              {error[r.id] && (
                <div className="text-[12px] text-muted" role="status">
                  {error[r.id]}
                </div>
              )}
              {!files[r.id] && !error[r.id] && (
                <div className="text-[12px] text-muted">Loading…</div>
              )}
              {Object.entries(files[r.id] || {}).map(([name, body]) => (
                <details key={name} open={name === "finding.md"}>
                  <summary className="text-[12.5px] text-muted cursor-pointer">{name}</summary>
                  {/* The finding and the scratchpad are written as markdown and meant to be
                      read — rendering them turns the sources into links you can actually
                      click. The transcript keeps its `--- Member A ---` rules and code-ish
                      shape, which markdown would mangle into headings. */}
                  {name === "transcript.md" ? (
                    <pre className="mt-2 text-[12px] leading-relaxed whitespace-pre-wrap break-words overflow-x-auto">
                      {body}
                    </pre>
                  ) : (
                    <div className="mt-2 text-[12px]">
                      <Markdown text={body} />
                    </div>
                  )}
                </details>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// -- shared -----------------------------------------------------------------------

function Toggle({
  label,
  help,
  checked,
  onChange,
}: {
  label: string;
  help?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-2.5 cursor-pointer">
      <input
        type="checkbox"
        className="mt-0.5 accent-accent"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>
        <span className="text-[12.5px] text-ink">{label}</span>
        {help && <span className={HELP + " block"}>{help}</span>}
      </span>
    </label>
  );
}
