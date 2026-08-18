import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { Attachment } from "../types";
import { isPdfFile, readFile } from "../attach";
import { getSettings, inspectPdf, renameSession } from "../api";
import { Dropdown, type Option } from "./Dropdown";
import { AddFolderForm } from "./AddFolderForm";
import { Icon } from "./Icon";
import { Toggle } from "./Toggle";
import { useRoots } from "../useRoots";
import {
  cancelDictation,
  getDictationLevel,
  getDictationStatus,
  isTauri,
  startDictation,
  stopDictation,
  type DictationStatus,
} from "../tauri";

// Plan + Custom hidden for this release (owner ask 2026-07-22): Plan's approval flow isn't
// polished enough to ship, and Custom (config.toml auto-allow rules) is a power-user mode
// with no in-app explanation. The server still honors both — a session already in one of
// those modes keeps working; the picker just doesn't offer them.
const PERMISSION_OPTIONS: Option[] = [
  { value: "discuss", label: "Discuss", description: "Chat and explore — no edits or commands" },
  { value: "interactive", label: "Ask for approval", description: "Ask before edits and commands" },
  { value: "auto", label: "Full access", description: "Run everything without asking" },
];

// Schedule helpers. A few presets plus the NATIVE datetime-local input covers "start this at
// 9pm" — no date-picker dependency, and no natural-language parsing in v1.
function fmtClock(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

function schedulePresets(): { label: string; at: Date }[] {
  const now = new Date();
  const tonight = new Date(now);
  tonight.setHours(21, 0, 0, 0);
  if (tonight <= now) tonight.setDate(tonight.getDate() + 1); // already past 9pm → tomorrow
  const tomorrow9 = new Date(now);
  tomorrow9.setDate(now.getDate() + 1);
  tomorrow9.setHours(9, 0, 0, 0);
  return [
    { label: "Tonight, 9:00 PM", at: tonight },
    { label: "In 1 hour", at: new Date(now.getTime() + 3600_000) },
    { label: "Tomorrow, 9:00 AM", at: tomorrow9 },
  ];
}

// No hardcoded model fallback: until the server supplies the list (a few seconds after a
// cold app boot), the picker renders a disabled "Loading models…" chip. A baked-in list
// goes stale and silently offers ids the backend never confirmed (caught 2026-07-21).

// Drop the provider prefix for display (anthropic:claude-opus-4-8 → claude-opus-4-8); full id on hover.
const shortModel = (m: string) => (m.includes(":") ? m.split(":").slice(1).join(":") : m);

// Identify an attachment by name + payload size so duplicates (e.g. the same file picked twice,
// or a prefill applied twice) collapse to one chip.
const attKey = (a: Attachment) =>
  a.kind === "text"
    ? `t:${a.name}:${a.text?.length ?? 0}`
    : `${a.kind[0]}:${a.name}:${a.data_url?.length ?? 0}`;
const mergeAttachments = (cur: Attachment[], add: Attachment[]): Attachment[] => {
  const seen = new Set(cur.map(attKey));
  return [...cur, ...add.filter((a) => !seen.has(attKey(a)))].slice(0, 8);
};

interface Props {
  sessionId: string;
  mode: string;
  model: string;
  models?: string[];
  modelLabels?: Record<string, string>; // curated display names (raw id when absent)
  // The model is FIXED once the session has history (§17): the picker renders ONLY on a fresh
  // session; after the first turn the fact lives in the topbar subtitle (§22) — no
  // interactive-then-disabled control.
  running: boolean;
  connected: boolean;
  // False when the default model's provider has no key — the composer shows a "connect a model"
  // banner and routes sends to setup (preserving the draft) instead of dropping them.
  modelReady?: boolean;
  onConnectModel?: () => void;
  onConfigureVoiceInput?: () => void;
  onSend: (text: string, attachments?: Attachment[]) => void;
  onSessionRenamed?: () => void;
  onInterrupt: () => void;
  // ⌘/Ctrl+Enter while a turn runs: inject into the live turn instead of queueing behind it.
  onSteer?: (text: string, attachments?: Attachment[]) => void;
  // Queue as a "not before" gate. `at` is epoch SECONDS.
  onSchedule?: (text: string, attachments: Attachment[], at: number) => void;
  onModeChange: (mode: string) => void;
  onModelChange: (model: string) => void;
  // When set (Code/Cowork), the Mode menu is shown. The folder/roots + branch controls left the
  // composer for the Session settings drawer (§22) — folder access is standing session config.
  workspace?: string;
  projectControlled?: boolean;
  // Unattended / send-approvals-to-Inbox — folded into the Mode menu (§22): "who approves, and
  // when" is one mental model. Absent handler = no toggle (e.g. Chat).
  unattended?: boolean;
  onUnattendedChange?: (on: boolean) => void;
  approvalSlot?: ReactNode;
  // Push text + attachments into the composer (e.g. a start-panel task card). The `nonce` makes
  // repeated identical prefills re-apply; the user can still edit before sending.
  prefill?: { text: string; attachments?: Attachment[]; nonce: number };
  // Changes when the active conversation changes; clears any unsent draft.
  resetKey?: string;
  // Surface-specific hint shown in the empty textarea.
  placeholder?: string;
}

// -- council mode ----------------------------------------------------------------------
// Sending "Council: <question>" is what convenes the panel, so the toggle prepends exactly
// that. No new plumbing through the send path, and the transcript shows the message the
// user actually sent rather than a hidden flag they cannot see or edit.
const COUNCIL_PREFIX = "Council:";

function councilKey(resetKey?: string): string {
  return `ocw.council.${resetKey || "default"}`;
}

function councilStored(resetKey?: string): boolean {
  try {
    return localStorage.getItem(councilKey(resetKey)) === "1";
  } catch {
    return false;
  }
}

/** The text as sent. Already-prefixed text is left alone — a user who types the word
 *  themselves with the toggle on must not get "Council: Council: …". */
export function withCouncil(text: string, on: boolean): string {
  const trimmed = text.trim();
  if (!on || !trimmed || trimmed.toLowerCase().startsWith(COUNCIL_PREFIX.toLowerCase())) {
    return trimmed;
  }
  return `${COUNCIL_PREFIX} ${trimmed}`;
}

export function Composer(props: Props) {
  // A first root can persist a fresh Chat session as Cowork server-side; local persona sync is out of scope here.
  const { addRoot } = useRoots(props.sessionId);
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [council, setCouncil] = useState(() => councilStored(props.resetKey));
  const [dragging, setDragging] = useState(false);
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);
  const [addingFolder, setAddingFolder] = useState(false);
  const [dictation, setDictation] = useState<DictationStatus | null>(null);
  const [dictationBusy, setDictationBusy] = useState<string | null>(null);
  const [dictationError, setDictationError] = useState<string | null>(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [attachNotice, setAttachNotice] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  // Armed schedule (epoch seconds) — Enter then schedules instead of sending. Null = off.
  const [scheduledFor, setScheduledFor] = useState<number | null>(null);
  const [scheduleOpen, setScheduleOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const noticeTimer = useRef<number | null>(null);

  // Rejected-attachment notice: visible ~8s, then clears (or on ✕).
  const showAttachNotice = (message: string) => {
    setAttachNotice(message);
    if (noticeTimer.current) window.clearTimeout(noticeTimer.current);
    noticeTimer.current = window.setTimeout(() => setAttachNotice(null), 8000);
  };

  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const max = parseFloat(getComputedStyle(el).lineHeight || "22") * 4;
    const next = Math.min(el.scrollHeight, max);
    el.style.height = `${Math.max(next, 24)}px`;
    el.style.overflowY = el.scrollHeight > max ? "auto" : "hidden";
  }, [text]);

  // Apply a prefill (text + attachments) pushed from outside, then focus the composer. Applied at
  // most once per nonce (a ref guards against StrictMode/re-render double-fires), and attachments
  // are de-duplicated so the same file never lands twice.
  const appliedNonce = useRef<number>(-1);
  useEffect(() => {
    const p = props.prefill;
    if (!p || p.nonce === appliedNonce.current) return;
    appliedNonce.current = p.nonce;
    setText(p.text);
    if (p.attachments?.length) setAttachments((cur) => mergeAttachments(cur, p.attachments!));
    textareaRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.prefill?.nonce]);

  // Clear the draft when the conversation changes, so a half-typed message / picked file doesn't
  // bleed from one session into another.
  useEffect(() => {
    setText("");
    setAttachments([]);
    setCouncil(councilStored(props.resetKey));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.resetKey]);

  // Council mode is per CONVERSATION, and it survives a reload: a chat opened to work
  // through one decision should still be a council chat tomorrow morning. Switching chats
  // reads the new one's setting rather than carrying the old one across.
  useEffect(() => {
    try {
      localStorage.setItem(councilKey(props.resetKey), council ? "1" : "0");
    } catch {
      // No storage — the toggle still works for this session.
    }
  }, [council, props.resetKey]);

  // Dictation is intentionally native-only: the browser/dev build remains a local server client
  // and never turns on the browser microphone or ships audio anywhere.
  useEffect(() => {
    if (!isTauri()) return;
    const refresh = (event?: Event) => {
      const supplied = (event as CustomEvent<DictationStatus> | undefined)?.detail;
      if (supplied) {
        setDictation(supplied);
        return;
      }
      void getDictationStatus().then((status) => status && setDictation(status));
    };
    refresh();
    window.addEventListener("coworker:voice-input-changed", refresh);
    return () => window.removeEventListener("coworker:voice-input-changed", refresh);
  }, []);

  useEffect(() => {
    if (!dictation?.recording) {
      setRecordingSeconds(0);
      return;
    }
    const started = Date.now();
    const timer = window.setInterval(() => {
      setRecordingSeconds(Math.floor((Date.now() - started) / 1000));
    }, 250);
    return () => window.clearInterval(timer);
  }, [dictation?.recording]);

  // Live waveform: poll mic loudness at ~10Hz while recording; the bars scroll left so the
  // trace reads as a real input meter (owner catch on DMG #28 — the first cut's bars were
  // decorative constants and read as fake).
  const [levels, setLevels] = useState<number[]>([]);
  useEffect(() => {
    if (!dictation?.recording) {
      setLevels([]);
      return;
    }
    const timer = window.setInterval(() => {
      getDictationLevel().then((level) => {
        if (typeof level === "number") setLevels((cur) => [...cur.slice(-13), level]);
      });
    }, 100);
    return () => window.clearInterval(timer);
  }, [dictation?.recording]);

  useEffect(() => {
    if (!dictation?.recording) return;
    const cancelOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      void cancelDictation()
        .catch(() => undefined)
        .finally(() => {
          void getDictationStatus().then((status) => status && setDictation(status));
        });
    };
    window.addEventListener("keydown", cancelOnEscape);
    return () => window.removeEventListener("keydown", cancelOnEscape);
  }, [dictation?.recording]);

  const voiceReady = !!dictation?.supported && !!dictation?.model_verified && !!dictation?.test_passed;
  const recordingTime = `${Math.floor(recordingSeconds / 60)}:${String(recordingSeconds % 60).padStart(2, "0")}`;

  // Attach-time PDF thresholds (Settings → Token savings): a PDF over the user's page or
  // size limit is REJECTED with a visible notice — never attached, never silently dropped.
  // The rationale is token cost: a big PDF re-rides every turn of the conversation.
  const addFiles = async (files: FileList | File[]) => {
    const list = Array.from(files);
    let maxPages = 20;
    let maxMb = 10;
    if (list.some(isPdfFile)) {
      try {
        const s = await getSettings();
        if (s.pdf_max_pages) maxPages = s.pdf_max_pages;
        if (s.pdf_max_mb) maxMb = s.pdf_max_mb;
      } catch {
        /* offline settings fetch — fall back to defaults */
      }
    }
    const accepted: File[] = [];
    for (const file of list) {
      if (isPdfFile(file) && file.size > maxMb * 1024 * 1024) {
        showAttachNotice(
          `${file.name} skipped — ${(file.size / 1024 / 1024).toFixed(1)} MB is over your ${maxMb} MB limit (Settings → Token savings)`,
        );
        continue;
      }
      accepted.push(file);
    }
    const read = (await Promise.all(accepted.map(readFile))).filter(Boolean) as Attachment[];
    const next: Attachment[] = [];
    for (const a of read) {
      if (a.kind === "pdf" && a.data_url) {
        const info = await inspectPdf(a.data_url).catch(() => null);
        if (info?.ok && (info.pages ?? 0) > maxPages) {
          showAttachNotice(
            `${a.name} skipped — ${info.pages} pages is over your ${maxPages}-page limit (Settings → Token savings)`,
          );
          continue;
        }
        if (info && !info.ok) {
          showAttachNotice(`${a.name} skipped — ${info.error || "could not read PDF"}`);
          continue;
        }
      }
      next.push(a);
    }
    if (next.length) setAttachments((a) => mergeAttachments(a, next));
  };

  // The "+" menu offers typed shortcuts; each just narrows the OS picker's filter.
  const pickFiles = (accept: string) => {
    setAttachMenuOpen(false);
    if (fileInput.current) {
      fileInput.current.accept = accept;
      fileInput.current.click();
    }
  };

  const needsModel = props.modelReady === false;

  // Enter queues while a turn runs (the server parks it), ⌘/Ctrl+Enter steers the live turn,
  // and an armed schedule turns Enter into "schedule for <time>". The composer is never locked.
  const submit = async (opts?: { steer?: boolean }) => {
    if (!opts?.steer) {
      const renameName = text.startsWith("/rename") && /^\/rename(?:\s|$)/.test(text)
        ? text.slice("/rename".length).trim()
        : /^\/\s/.test(text)
          ? text.slice(1).trim()
          : null;
      if (renameName !== null) {
        if (renaming) return;
        if (!renameName) {
          // Enter did nothing visible otherwise, which reads as the composer being stuck.
          showAttachNotice("Give the conversation a name: /rename <name>");
          return;
        }
        setRenaming(true);
        try {
          const result = await renameSession(props.sessionId, renameName);
          if (!result.ok) {
            showAttachNotice(result.error || "Conversation could not be renamed.");
            return;
          }
          // Only the command text goes; a rename is not a send, so staged attachments stay.
          setText("");
          showAttachNotice(`Conversation renamed to ${renameName}.`);
          props.onSessionRenamed?.();
        } catch (error) {
          showAttachNotice(
            error instanceof Error ? error.message : "Conversation could not be renamed.",
          );
        } finally {
          setRenaming(false);
        }
        return;
      }
    }
    // Steering injects into a turn that is already running, so a council prefix there would
    // convene a second panel mid-debate. Council mode applies to new messages only.
    const t = withCouncil(text, council && !(opts?.steer && props.running));
    if ((!t && attachments.length === 0) || dictation?.recording || dictationBusy) return;
    // No model connected: keep the draft (don't drop it) and send the user to setup instead.
    if (needsModel) {
      props.onConnectModel?.();
      return;
    }
    if (opts?.steer && props.running) {
      props.onSteer?.(t, attachments);
    } else if (scheduledFor) {
      props.onSchedule?.(t, attachments, scheduledFor);
      setScheduledFor(null);
    } else {
      // Server-side decision: runs now when idle, queues when busy. Keeping the branch on the
      // server means a turn ending mid-keystroke can't drop the prompt on the floor.
      props.onSend(t, attachments);
    }
    setText("");
    setAttachments([]);
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      void submit({ steer: props.running });
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void submit();
      return;
    }
    if (e.key === "Escape" && scheduleOpen) {
      e.preventDefault();
      setScheduleOpen(false);
    }
  };

  const onPaste = (e: React.ClipboardEvent) => {
    const imgs = Array.from(e.clipboardData.items)
      .filter((it) => it.kind === "file" && it.type.startsWith("image/"))
      .map((it) => it.getAsFile())
      .filter(Boolean) as File[];
    if (imgs.length) {
      e.preventDefault();
      addFiles(imgs);
    }
  };

  const toggleDictation = async () => {
    if (!isTauri() || dictationBusy) return;
    setDictationError(null);
    try {
      if (dictation?.recording) {
        setDictationBusy("Transcribing…");
        const transcript = await stopDictation();
        if (transcript === null) throw new Error("Could not transcribe your recording.");
        if (transcript.trim()) {
          setText((draft) => (draft.trim() ? `${draft.trimEnd()} ${transcript.trim()}` : transcript.trim()));
        }
        setDictation(await getDictationStatus());
        textareaRef.current?.focus();
        return;
      }

      const status = dictation || (await getDictationStatus());
      if (!status) throw new Error("Voice dictation is unavailable.");
      if (!status.supported || !status.model_verified || !status.test_passed) {
        props.onConfigureVoiceInput?.();
        return;
      }
      setDictationBusy("Starting microphone…");
      const recording = await startDictation();
      if (!recording?.recording) throw new Error("Could not start the microphone.");
      setDictation(recording);
    } catch (error) {
      setDictationError(error instanceof Error ? error.message : "Voice dictation is unavailable.");
      const status = await getDictationStatus();
      if (status) setDictation(status);
    } finally {
      setDictationBusy(null);
    }
  };

  const modelsLoaded = !!(props.models && props.models.length);
  const modelOptions: Option[] = Array.from(
    new Set([props.model, ...(props.models || [])]),
  ).map((m) => ({
    value: m,
    label: props.modelLabels?.[m] || shortModel(m),
  }));

  const iconBtn =
    "w-7 h-7 grid place-items-center rounded-md text-muted hover:text-ink hover:bg-paper shrink-0";

  // The send button is accent only when there's something to send — subtle grey otherwise, so the
  // composer isn't carrying a constant blue dot.
  const hasContent = text.trim().length > 0 || attachments.length > 0;

  return (
    <div className="composer-wrap px-6 pb-5 pt-4">
      {props.approvalSlot}

      {dictationError && (
        <div className="max-w-3xl mx-auto mb-2 px-1 text-[12px] text-red-600" role="alert">
          {dictationError}
        </div>
      )}

      {/* Rejected-attachment notice (PDF over the user's Token-savings thresholds). */}
      {attachNotice && (
        <div
          data-testid="attach-notice"
          className="max-w-3xl mx-auto mb-1.5 flex items-center gap-2 rounded-lg border border-warnInk/30 bg-warnSoft px-3 py-1.5 text-[12.5px] text-warnInk"
        >
          <span className="flex-1">{attachNotice}</span>
          <button
            className="shrink-0 opacity-60 hover:opacity-100"
            onClick={() => setAttachNotice(null)}
            title="Dismiss"
          >
            ✕
          </button>
        </div>
      )}

      {/* Attachments preview — a strip ABOVE the input box (mock/Claude-style). */}
      {attachments.length > 0 && (
        <div className="max-w-3xl mx-auto mb-1.5 flex flex-wrap gap-2">
          {attachments.map((a, i) => (
            <AttachChip key={i} a={a} onRemove={() => setAttachments((all) => all.filter((_, j) => j !== i))} />
          ))}
        </div>
      )}

      <div
        className={
          "composer max-w-3xl mx-auto rounded-2xl border border-line bg-panel shadow-sm" +
          (dragging ? " dragging" : "")
        }
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
        }}
      >
        <textarea
          ref={textareaRef}
          className="w-full block px-3.5 pt-3.5 pb-1.5 text-[14.5px]"
          placeholder={props.placeholder || "Ask the coworker…  (drop or paste files)"}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          onPaste={onPaste}
          rows={1}
        />

        {/* Three-control row (§22): + attach · Mode ⌄ …(right)… model (fresh only) · send */}
        <div className="px-2.5 pb-2.5 pt-1 flex items-center gap-1.5">
          {/* + attach menu */}
          <div className="relative">
            <button
              className={iconBtn + (attachMenuOpen ? " bg-paper text-ink" : "")}
              title="Attach"
              aria-label="Attach"
              onClick={() => setAttachMenuOpen((v) => !v)}
            >
              <Icon name="plus" size={17} />
            </button>
            {attachMenuOpen && (
              <>
                <div className="fixed inset-0 z-30" onClick={() => setAttachMenuOpen(false)} />
                <div className="absolute z-40 bottom-full mb-1 left-0 min-w-[180px] rounded-xl border border-line bg-panel shadow-2xl py-1.5">
                  {addingFolder ? (
                    <AddFolderForm
                      onAdd={async (path, writable) => {
                        const ok = await addRoot(path, writable);
                        if (ok !== false) setAttachMenuOpen(false);
                        return ok;
                      }}
                      startOpen
                      onDismiss={() => setAddingFolder(false)}
                    />
                  ) : (
                    <>
                      {attachItem("image", "Photo or image", () => pickFiles("image/*"))}
                      {attachItem("file", "PDF", () => pickFiles("application/pdf,.pdf"))}
                      {attachItem(
                        "fileCode",
                        "Other files",
                        () => pickFiles("text/*,.md,.csv,.json,.yaml,.yml,.log,.py,.ts,.tsx,.js,.rs,.go,.toml"),
                      )}
                      {props.workspace === undefined &&
                        !props.projectControlled &&
                        attachItem("folderPlus", "Folder…", () => setAddingFolder(true))}
                      {props.projectControlled && (
                        <div
                          className="flex items-center gap-2 px-2.5 py-2 text-[12.5px] text-faint"
                          data-testid="project-folder-source"
                        >
                          <Icon name="folder" size={14} className="shrink-0" />
                          <span>Working folder comes from project</span>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </>
            )}
          </div>
          <input
            ref={fileInput}
            type="file"
            multiple
            style={{ display: "none" }}
            onChange={(e) => {
              if (e.target.files) addFiles(e.target.files);
              e.target.value = "";
            }}
          />

          {/* Listening replaces the quiet middle controls with a LIVE waveform (mic RMS,
              polled ~10Hz, scrolling left) + elapsed time (§37). */}
          {dictation?.recording ? (
            <div className="voice-wave-row flex-1 flex items-center gap-2 ml-1" aria-hidden="true">
              <span className="voice-wave-line" />
              <span className="voice-wave-bars">
                {Array.from({ length: 14 }, (_, index) => {
                  const level = levels[levels.length - 14 + index] ?? 0;
                  return <i key={index} style={{ height: Math.round(4 + level * 24) }} />;
                })}
              </span>
              <span className="text-[12px] text-muted tabular-nums">{recordingTime}</span>
            </div>
          ) : props.workspace !== undefined ? (
            <ModeMenu
              mode={props.mode}
              onModeChange={props.onModeChange}
              unattended={props.unattended}
              onUnattendedChange={props.onUnattendedChange}
            />
          ) : null}

          {dictationBusy === "Transcribing…" && <span className="text-[11.5px] text-accent">Transcribing…</span>}

          <span className="ml-auto" />

          {/* model — a quiet chip, now for the session's whole life (§17 rev 2026-07-22:
              mid-session switching shipped, so the picker stays actionable; the topbar
              subtitle still states the current model). */}
          {!dictation?.recording && (needsModel ? (
            <button
              className="pill model-warn chip"
              onClick={() => props.onConnectModel?.()}
              title="Connect a model"
              aria-label="No model connected — connect a model"
            >
              <span className="pill-label">No model</span>
              <span className="model-warn-ico" aria-hidden>⚠</span>
            </button>
          ) : modelsLoaded ? (
            <Dropdown value={props.model} options={modelOptions} onChange={props.onModelChange} align="right" />
          ) : (
            <button
              className="pill chip text-faint cursor-default"
              disabled
              data-testid="models-loading"
              title="Fetching the model list from the server"
            >
              <span className="pill-label">Loading models…</span>
            </button>
          ))}

          {/* Council mode. Left of the mic so the two "how this message is handled" controls
              (which model, which panel) sit together, away from send. */}
          {!dictation?.recording && (
            <button
              className={"pill chip" + (council ? " is-on" : "")}
              onClick={() => setCouncil((on) => !on)}
              aria-pressed={council}
              data-testid="council-toggle"
              title={
                council
                  ? "Council is on: every message in this chat goes to the whole panel. Slower, and it spends every provider's credits. Click to turn off."
                  : "Council: put each message to every configured model, have them debate it, and get one finding back. Click to turn on."
              }
            >
              <span className="pill-dot" aria-hidden />
              <span className="pill-label">{council ? "Council on" : "Council"}</span>
            </button>
          )}

          {/* mic — immediately before send (owner call, DMG #28 walkthrough) */}
          {isTauri() && (
            <button
              className={
                iconBtn +
                (dictation?.recording ? " bg-red-50 text-red-600 hover:bg-red-100" : "") +
                (dictationBusy ? " opacity-60" : "") +
                (!voiceReady && !dictation?.recording ? " opacity-40" : "")
              }
              onClick={() => void toggleDictation()}
              disabled={!!dictationBusy}
              title={
                dictationBusy ||
                (dictation?.recording
                  ? "Stop recording and transcribe"
                  : voiceReady
                    ? "Start local voice dictation"
                    : "Configure Voice Input in Settings")
              }
              aria-label={dictation?.recording ? "Stop dictation" : voiceReady ? "Start dictation" : "Configure Voice Input in Settings"}
              aria-disabled={!voiceReady && !dictation?.recording}
            >
              <Icon name={dictation?.recording ? "stop" : "mic"} size={16} />
            </button>
          )}

          {/* schedule — a "not before" gate, so it also holds everything queued behind it */}
          <div className="relative">
            <button
              className={iconBtn + (scheduleOpen || scheduledFor ? " bg-paper text-ink" : "")}
              title={scheduledFor ? `Scheduled for ${fmtClock(scheduledFor)}` : "Send later"}
              aria-label="Schedule this prompt"
              aria-expanded={scheduleOpen}
              onClick={() => setScheduleOpen((v) => !v)}
            >
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" aria-hidden="true">
                <circle cx="12" cy="12" r="9" />
                <path d="M12 7v5l3 2" />
              </svg>
            </button>
            {scheduleOpen && (
              <>
                <div className="fixed inset-0 z-30" onClick={() => setScheduleOpen(false)} />
                <div className="absolute z-40 bottom-full mb-1 right-0 min-w-[220px] rounded-xl border border-line bg-panel shadow-2xl p-1.5">
                  {schedulePresets().map((p) => (
                    <button
                      key={p.label}
                      className="w-full text-left text-sm px-2.5 py-2 rounded-lg hover:bg-paper min-h-[40px]"
                      onClick={() => {
                        setScheduledFor(Math.floor(p.at.getTime() / 1000));
                        setScheduleOpen(false);
                      }}
                    >
                      {p.label}
                    </button>
                  ))}
                  <label className="block text-xs text-faint px-2.5 pt-2 pb-1">
                    Choose date and time
                  </label>
                  <input
                    type="datetime-local"
                    className="w-full text-sm bg-paper border border-line rounded-lg px-2 py-1.5 mb-1"
                    onChange={(e) => {
                      const v = e.target.value ? new Date(e.target.value) : null;
                      if (v && !Number.isNaN(v.getTime())) {
                        setScheduledFor(Math.floor(v.getTime() / 1000));
                        setScheduleOpen(false);
                      }
                    }}
                  />
                  {scheduledFor && (
                    <button
                      className="w-full text-left text-sm px-2.5 py-2 rounded-lg hover:bg-paper text-faint min-h-[40px]"
                      onClick={() => {
                        setScheduledFor(null);
                        setScheduleOpen(false);
                      }}
                    >
                      Clear schedule
                    </button>
                  )}
                </div>
              </>
            )}
          </div>

          {/* Stop keeps its own control — never overloaded onto submit (you may want both). */}
          {props.running && (
            <button className="btn danger" onClick={props.onInterrupt} title="Stop the running turn">
              ⏹ Stop
            </button>
          )}

          <button
            className={
              "h-7 px-2 rounded-full flex items-center gap-1 shrink-0 transition-colors active:scale-95 " +
              (hasContent && props.connected && !dictation?.recording && !dictationBusy
                ? "bg-accent text-white hover:brightness-105"
                : "bg-paper border border-line text-faint")
            }
            onClick={() => void submit()}
            disabled={!props.connected || !!dictation?.recording || !!dictationBusy || renaming}
            title={
              needsModel
                ? "Connect a model to send"
                : scheduledFor
                  ? `Schedule for ${fmtClock(scheduledFor)}`
                  : props.running
                    ? "Add to queue — ⌘Enter steers the running turn instead"
                    : "Send"
            }
            aria-label={scheduledFor ? "Schedule prompt" : props.running ? "Add to queue" : "Send"}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              {scheduledFor ? (
                <>
                  <circle cx="12" cy="12" r="9" />
                  <path d="M12 7v5l3 2" />
                </>
              ) : props.running ? (
                <path d="M4 6h11M4 12h11M4 18h7M18 15v6M15 18h6" />
              ) : (
                <path d="M12 19V5M5 12l7-7 7 7" />
              )}
            </svg>
            {(scheduledFor || props.running) && (
              <span className="text-xs pr-0.5">
                {scheduledFor ? fmtClock(scheduledFor) : "Queue"}
              </span>
            )}
          </button>
        </div>
      </div>
      <span className="sr-only" role="status" aria-live="polite">
        {dictation?.recording ? `Listening, ${recordingTime}` : dictationBusy || ""}
      </span>
    </div>
  );
}

// The composer's Mode menu (§22): a quiet "Mode ⌄" chip opening the five permission options with
// the current one marked, plus — when the session supports it — the "Send approvals to Inbox"
// toggle at the bottom (the old standalone InboxControl, folded in).
function ModeMenu({
  mode,
  onModeChange,
  unattended,
  onUnattendedChange,
}: {
  mode: string;
  onModeChange: (mode: string) => void;
  unattended?: boolean;
  onUnattendedChange?: (on: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const current = PERMISSION_OPTIONS.find((o) => o.value === mode);
  return (
    <div className="relative">
      {/* Borderless, and it names the CHOSEN mode (owner ask 2026-07-11, competitor composer
          comparison): "Ask for approval ⌄" not a generic "Mode ⌄" pill. aria-label stays
          "Mode" so the accessible name is stable across mode changes. */}
      <button
        className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[12px] text-muted hover:text-ink hover:bg-paper shrink-0"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Mode"
        title={
          `Mode: ${current?.label || mode}` +
          (unattended ? " · approvals go to the Inbox" : "")
        }
      >
        {current?.label || mode}
        <Icon name="chevronDown" size={11} className="text-faint" />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div
            className="absolute z-40 bottom-full mb-1 left-0 w-[260px] rounded-xl border border-line bg-panel shadow-2xl p-1.5"
            role="menu"
            data-testid="mode-menu"
          >
            {PERMISSION_OPTIONS.map((o) => (
              <button
                key={o.value}
                className="w-full flex flex-col items-start px-2.5 py-1.5 rounded-lg text-left hover:bg-paper"
                onClick={() => {
                  onModeChange(o.value);
                  setOpen(false);
                }}
              >
                <span
                  className={
                    "text-[13px] " + (o.value === mode ? "font-medium text-accent" : "text-ink")
                  }
                >
                  {o.label}
                  {o.value === mode && <span className="ml-1.5">✓</span>}
                </span>
                <span className="text-[11px] text-faint leading-snug">{o.description}</span>
              </button>
            ))}
            {onUnattendedChange && (
              <>
                <div className="my-1 border-t border-line" />
                <div className="flex items-center gap-2 px-2.5 py-1.5">
                  <span className="flex-1 min-w-0">
                    <span className="block text-[13px] text-ink">Send approvals to Inbox</span>
                    <span className="block text-[11px] text-faint leading-snug">
                      Approvals &amp; questions go to the Inbox; the agent keeps working.
                    </span>
                  </span>
                  <Toggle
                    checked={!!unattended}
                    onChange={onUnattendedChange}
                    title="Send approvals to the Inbox"
                  />
                </div>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// A row in the "+" attach menu.
function attachItem(icon: "image" | "file" | "fileCode" | "folderPlus", label: string, onClick: () => void) {
  return (
    <button
      className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[13px] text-left hover:bg-paper"
      onClick={onClick}
    >
      <Icon name={icon} size={15} className="shrink-0 text-muted" /> {label}
    </button>
  );
}

function AttachChip({ a, onRemove }: { a: Attachment; onRemove: () => void }) {
  return (
    <div className={"attach-chip" + (a.kind === "image" ? " img" : "")}>
      {a.kind === "image" ? (
        <img src={a.data_url} alt={a.name} />
      ) : (
        <>
          <Icon name="file" size={13} />
          <span className="attach-name">{a.name}</span>
        </>
      )}
      <button className="attach-x" onClick={onRemove} title="Remove">
        ✕
      </button>
    </div>
  );
}
