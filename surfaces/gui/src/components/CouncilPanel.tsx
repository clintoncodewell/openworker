import { useEffect, useState } from "react";
import { getCouncilLive, type CouncilLive } from "../api";
import { Icon } from "./Icon";

// A council blocks for minutes and returns nothing until it is finished, which reads as a
// hung app. The engine runs inside a tool call on a worker thread with no channel to the
// UI, so it writes its progress to a file and this polls it (GET /v1/council/live).
//
// Poll, not a socket: the state is a handful of lines that change a few times a minute, and
// a socket would be more moving parts than the thing it reports on.
const POLL_MS = 2500;
// How long after the last update a finished run stays visible. Long enough to read the
// notes after the answer lands, short enough that yesterday's run is not still sitting there.
const KEEP_MS = 10 * 60_000;
// A run that stops updating without reaching "done" crashed, or the sidecar was restarted
// under it. Nothing will ever clear the file, so an unfinished run that has gone quiet for
// longer than one member timeout is treated as gone rather than shown as debating forever.
const ABANDONED_MS = 6 * 60_000;

function short(model: string): string {
  // "azure:gpt-5.6-sol" → "gpt-5.6-sol". The vendor prefix is noise in a narrow column.
  return model.includes(":") ? model.slice(model.indexOf(":") + 1) : model;
}

export function useCouncilLive(active: boolean): CouncilLive | null {
  const [live, setLive] = useState<CouncilLive | null>(null);

  useEffect(() => {
    if (!active) return;
    let alive = true;
    const tick = () =>
      getCouncilLive()
        .then((next) => {
          if (!alive) return;
          const age = next?.updated ? Date.now() / 1000 - next.updated : 0;
          const limit = (next?.status === "done" ? KEEP_MS : ABANDONED_MS) / 1000;
          setLive(next && next.run && age <= limit ? next : null);
        })
        .catch(() => {
          // The sidecar restarting mid-poll is not worth showing anyone.
        });
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [active]);

  return live;
}

/** The debate as it happens: who is on the panel, where each one stands, and the notes they
 *  have posted to the shared scratchpad. */
export function CouncilPanel({ live }: { live: CouncilLive }) {
  const done = live.status === "done";
  const stances = live.stances || [];
  const notes = live.notes || [];
  const failed = stances.filter((s) => s.error);

  return (
    <div className="council-live" data-testid="council-live">
      <div className="council-status">
        {!done && <span className="council-spinner" aria-hidden />}
        <span>{done ? "Finding ready" : live.status || "starting"}</span>
        {!done && !!live.rounds && (
          <span className="council-round">
            round {live.round || 1} of {live.rounds}
          </span>
        )}
      </div>

      {!!live.question && <div className="council-question">{live.question}</div>}

      {!!(live.queries || []).length && (
        <div className="council-queries">
          <Icon name="search" size={12} />
          <span>{(live.queries || []).join(" · ")}</span>
        </div>
      )}

      {(live.panel || []).length > 0 && (
        <ul className="council-members">
          {(live.panel || []).map((m) => {
            const said = stances.find((s) => s.model === m.model);
            return (
              <li key={m.model} className={said?.error ? "is-failed" : said ? "is-done" : "is-waiting"}>
                <span className="council-member-name">
                  {short(m.model)}
                  <span className="council-member-role">{m.role}</span>
                </span>
                <span className="council-member-stance">
                  {said?.error
                    ? "no answer"
                    : said?.stance
                      ? `${said.stance}${said.confidence ? ` · ${said.confidence}` : ""}`
                      : "thinking…"}
                </span>
              </li>
            );
          })}
        </ul>
      )}

      {notes.length > 0 && (
        <div className="council-notes">
          <div className="council-notes-head">Shared scratchpad</div>
          {notes.map((n, i) => (
            <div className="council-note" key={i}>
              <b>{short(n.model)}</b> {n.note}
            </div>
          ))}
        </div>
      )}

      {done && failed.length > 0 && (
        <div className="council-warn">
          {failed.length} member{failed.length > 1 ? "s" : ""} did not answer:{" "}
          {failed.map((f) => short(f.model)).join(", ")}
        </div>
      )}
      {done &&
        ((live.report?.notes as string[] | undefined) || []).map((note, i) => (
          <div className="council-warn" key={i}>
            {note}
          </div>
        ))}
    </div>
  );
}
