import { useEffect, useState } from "react";

// The "Next up" shelf: prompts submitted while a turn is running wait here and run in order.
// Anchored directly above the composer (not in the transcript) so it never scrolls away —
// queued work is a control surface, not conversation.
//
// Two distinct actions, kept legible: an item WAITS its turn, or you "Steer now" to inject it
// into the running turn. A steer lands at the agent's next step, so nothing here claims it
// applied instantly.

export type QueueItem = {
  id: string;
  text: string;
  attachments?: unknown[];
  created_at?: number;
  not_before?: number | null; // epoch seconds; a scheduled "not before" gate
};

function fmtWhen(epochSeconds: number): string {
  const d = new Date(epochSeconds * 1000);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  const time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (sameDay) return time;
  const tomorrow = new Date(now.getTime() + 86400000);
  if (d.toDateString() === tomorrow.toDateString()) return `tomorrow ${time}`;
  return `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })} ${time}`;
}

export function QueueShelf(props: {
  items: QueueItem[];
  paused: boolean;
  running: boolean;
  onSteerNow: (id: string) => void;
  onRemove: (id: string) => void;
  onEdit: (id: string, text: string) => void;
  onReorder: (order: string[]) => void;
  onResume: () => void;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [collapsed, setCollapsed] = useState(false);
  const [pending, setPending] = useState<Set<string>>(() => new Set());

  // Every server snapshot acknowledges all outstanding commands. Until then, disable the
  // affected control so a double-click cannot send duplicate promote/remove/reorder frames.
  useEffect(() => setPending(new Set()), [props.items, props.paused]);

  // Empty queue renders nothing at all — no placeholder taking up space.
  if (!props.items.length) return null;

  const scheduled = props.items.filter((i) => i.not_before != null).length;

  const act = (key: string, fn: () => void) => {
    if (pending.has(key)) return;
    setPending((current) => new Set(current).add(key));
    fn();
  };

  const move = (index: number, delta: number) => {
    const next = props.items.slice();
    const target = index + delta;
    if (target < 0 || target >= next.length) return;
    const [item] = next.splice(index, 1);
    next.splice(target, 0, item);
    act("reorder", () => props.onReorder(next.map((i) => i.id)));
  };

  const startEdit = (item: QueueItem) => {
    setEditing(item.id);
    setDraft(item.text);
  };

  const commitEdit = (id: string) => {
    const t = draft.trim();
    if (t) act(id, () => props.onEdit(id, t));
    setEditing(null);
  };

  return (
    <div className="shrink-0 px-6">
      <section
        aria-label="Pending prompts"
        className="mx-auto mb-2 w-full max-w-3xl overflow-hidden rounded-xl border border-line bg-panel/80 backdrop-blur"
      >
      <header className="flex items-center gap-2 px-3 py-2 border-b border-line/60">
        <span className="text-xs font-medium text-ink">Next up</span>
        <span className="text-xs text-faint tabular-nums">
          {props.items.length} pending{scheduled ? ` · ${scheduled} scheduled` : ""}
        </span>
        <div className="flex-1" />
        {props.paused && (
          <button
            className="btn text-xs h-7 px-2"
            onClick={() => act("resume", props.onResume)}
            disabled={pending.has("resume")}
            title="Resume running queued prompts"
          >
            Resume
          </button>
        )}
        <button
          className="text-xs text-faint hover:text-ink px-2 h-7 rounded-lg"
          onClick={() => setCollapsed((c) => !c)}
          aria-expanded={!collapsed}
          title={collapsed ? "Show queued prompts" : "Hide queued prompts"}
        >
          {collapsed ? "Show" : "Hide"}
        </button>
      </header>

      {props.paused && (
        <p className="px-3 py-1.5 text-xs text-amber-500 border-b border-line/60">
          Paused — the last turn didn’t finish cleanly, so nothing runs until you resume.
        </p>
      )}

      {!collapsed && (
        <ol className="max-h-[320px] overflow-y-auto">
          {props.items.map((item, i) => (
            <li
              key={item.id}
              className="flex items-start gap-2 px-3 py-2 border-b border-line/40 last:border-b-0"
            >
              <span className="text-xs text-faint tabular-nums pt-1.5 w-4 shrink-0">{i + 1}</span>

              <div className="flex-1 min-w-0">
                {editing === item.id ? (
                  <textarea
                    className="w-full text-sm bg-paper border border-line rounded-lg p-2"
                    value={draft}
                    autoFocus
                    rows={3}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                        e.preventDefault();
                        commitEdit(item.id);
                      } else if (e.key === "Escape") {
                        e.preventDefault();
                        setEditing(null);
                      }
                    }}
                  />
                ) : (
                  <p className="text-sm text-ink line-clamp-2 [text-wrap:pretty]">{item.text}</p>
                )}
                {item.not_before != null ? (
                  <span className="mt-1 inline-flex items-center gap-1 text-xs text-violet-400">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                      <circle cx="12" cy="12" r="9" />
                      <path d="M12 7v5l3 2" />
                    </svg>
                    Not before {fmtWhen(item.not_before)}
                  </span>
                ) : null}
                {(item.attachments?.length ?? 0) > 0 && (
                  <span className="mt-1 ml-2 inline-flex text-xs text-faint">
                    {item.attachments!.length} attachment{item.attachments!.length === 1 ? "" : "s"}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-0.5 shrink-0">
                {editing === item.id ? (
                    <button
                      className="btn text-xs h-9 px-2"
                      onClick={() => commitEdit(item.id)}
                      disabled={pending.has(item.id)}
                    >
                    Save
                  </button>
                ) : (
                  <>
                    {props.running && (
                      <button
                        className="text-xs h-9 px-2 rounded-lg text-cyan-400 hover:bg-paper"
                        onClick={() => act(item.id, () => props.onSteerNow(item.id))}
                        disabled={pending.has(item.id)}
                        title="Send now into the running turn — lands at the agent’s next step"
                      >
                        Steer now
                      </button>
                    )}
                    <button
                      className="w-9 h-9 grid place-items-center rounded-lg text-faint hover:text-ink hover:bg-paper"
                      onClick={() => move(i, -1)}
                      disabled={i === 0 || pending.has("reorder")}
                      title="Move up"
                      aria-label={`Move item ${i + 1} up`}
                    >
                      ↑
                    </button>
                    <button
                      className="w-9 h-9 grid place-items-center rounded-lg text-faint hover:text-ink hover:bg-paper"
                      onClick={() => move(i, 1)}
                      disabled={i === props.items.length - 1 || pending.has("reorder")}
                      title="Move down"
                      aria-label={`Move item ${i + 1} down`}
                    >
                      ↓
                    </button>
                    <button
                      className="w-9 h-9 grid place-items-center rounded-lg text-faint hover:text-ink hover:bg-paper"
                      onClick={() => startEdit(item)}
                      disabled={pending.has(item.id)}
                      title="Edit"
                      aria-label={`Edit item ${i + 1}`}
                    >
                      ✎
                    </button>
                    <button
                      className="w-9 h-9 grid place-items-center rounded-lg text-faint hover:text-red-400 hover:bg-paper"
                      onClick={() => act(item.id, () => props.onRemove(item.id))}
                      disabled={pending.has(item.id)}
                      title="Remove"
                      aria-label={`Remove item ${i + 1}`}
                    >
                      ✕
                    </button>
                  </>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}

      <span className="sr-only" role="status" aria-live="polite">
        {props.items.length} prompt{props.items.length === 1 ? "" : "s"} queued
        {props.paused ? ", queue paused" : ""}
      </span>
      </section>
    </div>
  );
}
