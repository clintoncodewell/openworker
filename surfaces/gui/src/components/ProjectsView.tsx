import { useEffect, useMemo, useState } from "react";
import {
  addProjectSession,
  announceProjectsChanged,
  createProject,
  deleteProject,
  getProject,
  getProjects,
  refreshProject,
  updateProject,
  type Project,
  type ProjectMutationResult,
} from "../api";
import type { SessionInfo } from "../types";
import { Icon } from "./Icon";
import { Markdown } from "./Markdown";

const CARD = "rounded-xl2 border border-line bg-panel";
const INPUT =
  "w-full min-w-0 px-3 py-2 rounded-lg border border-line bg-paper text-[13px] text-ink outline-none focus:border-accent";
// `active:scale-[0.96]` is the press signifier — without it a click on a local control
// gives no feedback at all until its network round trip returns, which on a slow call reads
// as the click having missed.
const PRESS = "transition-transform active:scale-[0.96] disabled:active:scale-100";
const BTN =
  "text-[12.5px] px-3 py-2 rounded-lg border border-line bg-paper hover:border-lineStrong shrink-0 disabled:opacity-40 " +
  PRESS;
const BTN_ACCENT =
  "text-[12.5px] px-3 py-2 rounded-lg bg-accent text-white shrink-0 disabled:opacity-40 " + PRESS;
const HELP = "text-[12px] text-muted mt-1.5 leading-relaxed";

type ProjectSections = {
  purpose: string;
  whereItStands: string;
  decisions: string;
  openThreads: string;
};

// The one test for "is this a project, or an {ok:false} error payload?". Used by the
// mutation guard AND by the list loader: getProject RESOLVES on a deleted project, so
// Promise.allSettled's fulfilled-filter lets the error through and the malformed entry
// only blows up later, when the card is clicked.
function isProject(value: unknown): value is Project {
  return !!value && typeof (value as Project).project_md === "string";
}

function projectSections(markdown: string): ProjectSections {
  const sections = new Map<string, string>();
  const headings = [...markdown.matchAll(/^##\s+(.+?)\s*$/gm)];
  headings.forEach((heading, index) => {
    const start = (heading.index || 0) + heading[0].length;
    const end = headings[index + 1]?.index ?? markdown.length;
    sections.set(heading[1].trim().toLowerCase(), markdown.slice(start, end).trim());
  });
  return {
    purpose: sections.get("purpose") || "",
    whereItStands: sections.get("where it stands") || "",
    decisions: sections.get("decisions") || "",
    openThreads: sections.get("open threads") || "",
  };
}

function summarySentence(markdown: string): string {
  const summary = projectSections(markdown).whereItStands
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/^\s*[-*+]\s+(?:\[[ xX]\]\s*)?/gm, "")
    .replace(/[*_`>#]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!summary) return "";
  return summary.match(/^.*?[.!?](?=\s|$)/)?.[0] || summary;
}

function relativeTime(value: string): string {
  const then = Date.parse(value);
  if (Number.isNaN(then)) return "recently";
  const seconds = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} day${days === 1 ? "" : "s"} ago`;
  return new Date(then).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main className="flex-1 min-w-0 flex bg-paper">
      <div className="flex-1 min-w-0 overflow-y-auto hairline-scroll">
        <div className="max-w-4xl mx-auto px-7 py-6">{children}</div>
      </div>
    </main>
  );
}

interface Props {
  recentSessions: SessionInfo[];
  onOpenSession: (id: string, workspace: string, agent: string) => void;
  onNewConversation: (projectId: string) => Promise<void> | void;
}

export function ProjectsView({ recentSessions, onOpenSession, onNewConversation }: Props) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [openProject, setOpenProject] = useState<Project | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadFailed, setLoadFailed] = useState(false);

  const load = async () => {
    setLoading(true);
    setLoadFailed(false);
    try {
      const summaries = await getProjects();
      const settled = await Promise.allSettled(summaries.map((project) => getProject(project.id)));
      const full = settled
        .filter((result): result is PromiseFulfilledResult<Project> => result.status === "fulfilled")
        .map((result) => result.value)
        .filter(isProject)
        .sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at));
      setProjects(full);
    } catch {
      setProjects([]);
      setLoadFailed(true);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  if (openProject) {
    return (
      <ProjectDetail
        project={openProject}
        recentSessions={recentSessions}
        onChange={setOpenProject}
        onOpenSession={onOpenSession}
        onNewConversation={onNewConversation}
        onBack={() => {
          setOpenProject(null);
          load();
        }}
        onDelete={() => {
          setOpenProject(null);
          load();
        }}
      />
    );
  }

  const create = async (name: string, purpose: string) => {
    const project = await createProject({ name, purpose });
    announceProjectsChanged();
    setShowForm(false);
    setOpenProject(project);
  };

  return (
    <Shell>
      <div className="flex items-start gap-3 mb-4">
        <h2 className="flex-1 min-w-0 text-[18px] font-semibold tracking-tight">Projects</h2>
        <button className={BTN_ACCENT} onClick={() => setShowForm((value) => !value)}>
          New project
        </button>
      </div>

      {showForm && <NewProjectForm onCancel={() => setShowForm(false)} onCreate={create} />}

      {loading ? (
        <div className="text-[13px] text-muted">Loading…</div>
      ) : loadFailed ? (
        <div className="text-[13px] text-muted" role="status">Projects could not be loaded.</div>
      ) : projects.length === 0 ? (
        <div className={CARD + " p-5 max-w-2xl"}>
          <p className="text-[13px] text-muted leading-relaxed">
            A project keeps one topic together: its conversations, its files, and a brief that keeps
            itself up to date. Start one and every chat you put in it feeds the same picture.
          </p>
          <button className={BTN_ACCENT + " mt-4"} onClick={() => setShowForm(true)}>
            New project
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-2.5">
          {projects.map((project) => {
            const summary = summarySentence(project.project_md);
            const count = project.session_ids.length;
            const conversations =
              count === 0
                ? "No conversations yet"
                : `${count} conversation${count === 1 ? "" : "s"}`;
            return (
              <button
                key={project.id}
                className={CARD + " px-4 py-3 text-left hover:border-lineStrong transition-colors"}
                onClick={() => setOpenProject(project)}
              >
                <div className="text-[13.5px] font-semibold">{project.name}</div>
                <div className="text-[12.5px] text-muted mt-1 leading-relaxed text-pretty">
                  {summary || "No summary yet — it fills in after the first conversation."}
                </div>
                <div className="text-[11.5px] text-faint mt-2 tabular-nums">
                  {conversations} · updated {relativeTime(project.updated_at)}
                </div>
              </button>
            );
          })}
        </div>
      )}
    </Shell>
  );
}

function NewProjectForm({
  onCancel,
  onCreate,
}: {
  onCancel: () => void;
  onCreate: (name: string, purpose: string) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [purpose, setPurpose] = useState("");
  const [creating, setCreating] = useState(false);

  const submit = async () => {
    if (!name.trim()) return;
    setCreating(true);
    try {
      await onCreate(name.trim(), purpose.trim());
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className={CARD + " p-4 mb-4 max-w-2xl"}>
      <div className="grid gap-3">
        <label className="block">
          <span className="text-[12.5px] font-medium text-ink">Name</span>
          <input
            className={INPUT + " mt-1.5"}
            value={name}
            autoFocus
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <label className="block">
          <span className="text-[12.5px] font-medium text-ink">What is this project for?</span>
          <input
            className={INPUT + " mt-1.5"}
            value={purpose}
            onChange={(event) => setPurpose(event.target.value)}
          />
        </label>
      </div>
      <div className="flex items-center gap-2 mt-4">
        <button className={BTN_ACCENT} disabled={!name.trim() || creating} onClick={submit}>
          {creating ? "Creating…" : "Create project"}
        </button>
        <button className={BTN} disabled={creating} onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}

function ProjectDetail({
  project,
  recentSessions,
  onChange,
  onOpenSession,
  onNewConversation,
  onBack,
  onDelete,
}: {
  project: Project;
  recentSessions: SessionInfo[];
  onChange: (project: Project) => void;
  onOpenSession: (id: string, workspace: string, agent: string) => void;
  onNewConversation: (projectId: string) => Promise<void> | void;
  onBack: () => void;
  onDelete: () => void;
}) {
  const sections = projectSections(project.project_md);
  const [editingName, setEditingName] = useState(false);
  const [name, setName] = useState(project.name);
  const [purpose, setPurpose] = useState(sections.purpose);
  const [instructions, setInstructions] = useState(project.instructions || "");
  const [refreshing, setRefreshing] = useState(false);
  const [deleteArmed, setDeleteArmed] = useState(false);
  const [selectedSession, setSelectedSession] = useState("");
  const [addingSession, setAddingSession] = useState(false);
  const [startingConversation, setStartingConversation] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setName(project.name);
    setPurpose(projectSections(project.project_md).purpose);
    setInstructions(project.instructions || "");
  }, [project]);

  const unattached = useMemo(() => {
    const attached = new Set(project.session_ids);
    return recentSessions.filter(
      (session) => !attached.has(session.session_id) && !session.session_id.startsWith("__"),
    );
  }, [project.session_ids, recentSessions]);

  const showFailure = (value: unknown) => {
    setError(value instanceof Error ? value.message : "Project could not be updated.");
  };

  const applyProject = (next: ProjectMutationResult) => {
    if (!isProject(next)) {
      const message =
        next && typeof next === "object" && "error" in next && typeof next.error === "string"
          ? next.error
          : "Project could not be updated.";
      setError(message);
      return false;
    }
    setError("");
    onChange(next as Project);
    return true;
  };

  const saveName = async () => {
    const next = name.trim();
    setEditingName(false);
    if (!next) {
      setName(project.name);
      return;
    }
    if (next === project.name) return;
    try {
      applyProject(await updateProject(project.id, { name: next }));
    } catch (failure) {
      showFailure(failure);
    }
  };

  const savePurpose = async () => {
    const next = purpose.trim();
    if (next === sections.purpose) return;
    try {
      applyProject(await updateProject(project.id, { purpose: next }));
    } catch (failure) {
      showFailure(failure);
    }
  };

  const saveInstructions = async () => {
    const next = instructions.trim();
    if (next === (project.instructions || "")) return;
    try {
      applyProject(await updateProject(project.id, { instructions: next }));
    } catch (failure) {
      showFailure(failure);
    }
  };

  const refresh = async () => {
    setRefreshing(true);
    try {
      applyProject(await refreshProject(project.id));
    } catch (failure) {
      showFailure(failure);
    } finally {
      setRefreshing(false);
    }
  };

  const addSession = async () => {
    if (!selectedSession) return;
    setAddingSession(true);
    try {
      if (applyProject(await addProjectSession(project.id, selectedSession))) {
        setSelectedSession("");
      }
    } catch (failure) {
      showFailure(failure);
    } finally {
      setAddingSession(false);
    }
  };

  const startConversation = async () => {
    if (startingConversation) return;
    setStartingConversation(true);
    setError("");
    try {
      await onNewConversation(project.id);
    } catch (failure) {
      showFailure(failure);
      setStartingConversation(false);
    }
  };

  const remove = async () => {
    if (!deleteArmed) {
      setDeleteArmed(true);
      return;
    }
    await deleteProject(project.id);
    announceProjectsChanged();
    onDelete();
  };

  return (
    <Shell>
      <button className="text-[13px] text-muted hover:text-ink mb-3" onClick={onBack}>
        ← Projects
      </button>

      <div className="mb-5">
        {editingName ? (
          <input
            className={INPUT + " max-w-xl text-[18px] font-semibold"}
            aria-label="Project name"
            value={name}
            autoFocus
            onChange={(event) => setName(event.target.value)}
            onBlur={saveName}
            onKeyDown={(event) => {
              if (event.key === "Enter") event.currentTarget.blur();
              if (event.key === "Escape") {
                setName(project.name);
                setEditingName(false);
              }
            }}
          />
        ) : (
          <button className="text-left" onClick={() => setEditingName(true)}>
            <h2 className="text-[18px] font-semibold tracking-tight">{project.name}</h2>
          </button>
        )}
        {error && (
          <p className="mt-2 text-[12px] text-danger" role="alert">
            {error}
          </p>
        )}
      </div>

      <div className="grid gap-4">
        <section className={CARD + " p-4"}>
          <label className="block">
            <span className="text-[13px] font-semibold text-ink">Purpose</span>
            <textarea
              className={INPUT + " mt-2 min-h-[96px] resize-y"}
              value={purpose}
              onChange={(event) => setPurpose(event.target.value)}
              onBlur={savePurpose}
            />
          </label>
          <p className={HELP}>Yours. Nothing rewrites this.</p>
        </section>

        <section>
          <div className="flex items-center justify-between gap-3 mb-2">
            <p className="text-[12px] text-muted">Kept up to date after each conversation.</p>
            <button
              className="text-[12.5px] text-muted hover:text-ink disabled:opacity-50"
              disabled={refreshing}
              onClick={refresh}
            >
              {refreshing ? "Refreshing…" : "Refresh now"}
            </button>
          </div>
          <div className="grid gap-3">
            <BriefSection title="Where it stands" markdown={sections.whereItStands} />
            <BriefSection title="Decisions" markdown={sections.decisions} />
            <BriefSection title="Open threads" markdown={sections.openThreads} />
          </div>
        </section>

        <section className={CARD + " p-4"}>
          <h3 className="text-[13px] font-semibold">Conversations</h3>
          <button
            className={BTN_ACCENT + " mt-3 inline-flex items-center gap-2 hover:opacity-90"}
            disabled={startingConversation}
            onClick={startConversation}
          >
            <Icon name="plus" size={14} className="shrink-0" />
            {startingConversation ? "Starting…" : "New conversation"}
          </button>
          {project.sessions.length === 0 ? (
            <p className={HELP}>No conversations yet.</p>
          ) : (
            <div className="mt-2 divide-y divide-line">
              {project.sessions.map((session) => (
                <button
                  key={session.session_id}
                  className="w-full text-left py-2 text-[12.5px] text-ink hover:text-accent"
                  onClick={() => onOpenSession(session.session_id, session.workspace, session.agent)}
                >
                  {session.title || "Untitled conversation"}
                </button>
              ))}
            </div>
          )}
          <div className="mt-4 pt-3 border-t border-line">
            <p className="text-[11.5px] text-muted mb-2">Or add an existing conversation</p>
            <div className="flex gap-2">
              <select
                className={INPUT}
                aria-label="Recent conversation"
                value={selectedSession}
                onChange={(event) => setSelectedSession(event.target.value)}
              >
                <option value="">Choose a recent conversation</option>
                {unattached.map((session) => (
                  <option key={session.session_id} value={session.session_id}>
                    {session.title || "Untitled conversation"}
                  </option>
                ))}
              </select>
              <button className={BTN} disabled={!selectedSession || addingSession} onClick={addSession}>
                {addingSession ? "Adding…" : "Add"}
              </button>
            </div>
          </div>
        </section>

        <section className={CARD + " p-4"}>
          <label className="block">
            <span className="text-[13px] font-semibold text-ink">Instructions</span>
            <textarea
              className={INPUT + " mt-2 min-h-[112px] resize-y"}
              value={instructions}
              onChange={(event) => setInstructions(event.target.value)}
              onBlur={saveInstructions}
            />
          </label>
          <p className={HELP}>Told to the assistant in every conversation in this project.</p>
        </section>

        <section className={CARD + " p-4"}>
          <h3 className="text-[13px] font-semibold">Files</h3>
          {project.files.length === 0 ? (
            <p className={HELP}>
              Drop files into the project folder to give every conversation the same reference material.
            </p>
          ) : (
            <ul className="mt-2 space-y-1.5 text-[12.5px] text-muted">
              {project.files.map((file) => <li key={file}>{file}</li>)}
            </ul>
          )}
        </section>
      </div>

      <button
        className={
          "text-[12.5px] px-3 py-2.5 -ml-3 rounded-lg mt-6 " +  // padding buys a real hit area
          PRESS +
          (deleteArmed
            ? " text-danger font-medium bg-dangerSoft"
            : " text-muted hover:text-danger")
        }
        onClick={remove}
      >
        {deleteArmed ? "Really delete?" : "Delete project"}
      </button>
    </Shell>
  );
}

function BriefSection({ title, markdown }: { title: string; markdown: string }) {
  return (
    <section className={CARD + " p-4"}>
      <h3 className="text-[13px] font-semibold mb-2">{title}</h3>
      {markdown ? <Markdown text={markdown} /> : <p className="text-[12.5px] text-muted">Nothing here yet.</p>}
    </section>
  );
}
