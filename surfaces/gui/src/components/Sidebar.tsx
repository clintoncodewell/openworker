import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  announceCloudChanged,
  AUTOMATIONS_CHANGED,
  CLOUD_CHANGED,
  cloudLogin,
  cloudLogout,
  createFolder,
  getAutomations,
  getCloudStatus,
  getPersonas,
  getProjects,
  getSettings,
  listFolders,
  proposeMagicSort,
  INBOX_UNLOCK,
  PERSONAS_CHANGED,
  PROJECTS_CHANGED,
  renameFolder,
  setNavLayout,
  waitForCloudSignIn,
  type Automation,
  type CloudStatus,
  type MagicSortApplyResult,
  type MagicSortProposal,
  type Persona,
  type ProjectSummary,
  type RecentWorkspace,
  type SurfaceVisibility,
} from "../api";
import type { ChatFolder, SessionInfo } from "../types";
import { isProjectScoped, shortPersonaName } from "../personaScope";
import { ConnectorIcon } from "../connectors/ConnectorIcon";
import { Icon, type IconName } from "./Icon";
import { PersonaGlyph, personaGlyph } from "./personaIcon";
import { SearchModal } from "./SearchModal";
import { baseName } from "../paths";
import { showPersonas } from "../flags";

// Session surfaces shown as accordions, in display order. The surfaced personas drive this list
// (so third-party / Ops personas appear); the hardcoded set is the fallback before personas load.
const SURFACES: { key: string; label: string; icon: IconName; cls: string }[] = [
  { key: "cowork", label: "Coworker", icon: "diamond", cls: "ico-cowork" },
  { key: "chat", label: "Chat", icon: "chat", cls: "ico-chat" },
  { key: "code", label: "Code", icon: "code", cls: "ico-code" },
];

const surfaceFromPersona = (p: Persona) => ({
  key: p.id,
  label: shortPersonaName(p.name, p.id),
  icon: personaGlyph(p.icon, p.family),
  cls: `ico-${p.icon || "cowork"}`,
});

// Attention = Inbox items awaiting a session (an accent count that bubbles session → persona →
// footer Inbox — all views of the one Inbox queue, never a second list).
function AttnBadge({ n }: { n: number }) {
  if (!n) return null;
  return (
    <span
      className="text-[10px] font-semibold text-ink bg-faint/30 rounded-full px-1.5 leading-[15px] shrink-0"
      title={`${n} awaiting your attention`}
    >
      {n > 99 ? "99+" : n}
    </span>
  );
}

// UX-023: unseen-run count on a Scheduled entry. Deliberately QUIET — same neutral
// treatment as the attention badge; failure only colors the tooltip's words, not the
// sidebar (owner call 2026-07-20: no color, and the entry alone carries the count).
function UnseenBadge({ n, failed }: { n: number; failed?: boolean }) {
  if (!n) return null;
  return (
    <span
      className="text-[10px] font-semibold text-ink bg-faint/30 rounded-full px-1.5 leading-[15px] shrink-0"
      title={failed ? `${n} new run${n > 1 ? "s" : ""} — the latest failed` : `${n} new run${n > 1 ? "s" : ""}`}
    >
      {n > 99 ? "99+" : n}
    </span>
  );
}

function ProjectCountBadge({ n }: { n: number }) {
  if (!n) return null;
  return (
    <span
      className="text-[10px] font-semibold text-ink bg-faint/30 rounded-full px-1.5 leading-[15px] shrink-0"
      title={`${n} project${n === 1 ? "" : "s"}`}
    >
      {n > 99 ? "99+" : n}
    </span>
  );
}

// Liveness = working (in-flight turn) / sleeping (a self-wake is pending). A count-less dot that
// never bubbles — it says "this is alive", not "this needs you".
function LiveDot({ state }: { state?: "working" | "sleeping" | "idle" }) {
  if (state !== "working" && state !== "sleeping") return null;
  return state === "working" ? (
    <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse shrink-0" title="Working now" />
  ) : (
    <span
      className="w-1.5 h-1.5 rounded-full bg-faint/60 shrink-0"
      title="Sleeping (will wake itself)"
    />
  );
}

// §31: a session spawned by a platform mention wears its platform's logo, right-aligned beside
// the title cluster (owner call 2026-07-13). Slack today; the origin key is the platform id.
function OriginIcon({ s }: { s: SessionInfo }) {
  if (s.origin !== "slack") return null;
  return (
    <ConnectorIcon
      connector={{ logo: "slack", brand_color: "#611f69" }}
      size={12}
      title={s.origin_label || "From Slack"}
    />
  );
}

// A subscribed-connector presence dot (right edge of a row). Brand-colorless here — the sidebar
// isn't passed the connector registry — so it reads as a neutral "listening on a channel" dot.
function ConnectorDot({ subs }: { subs?: string[] }) {
  if (!subs || subs.length === 0) return null;
  return (
    <span
      className="w-1.5 h-1.5 rounded-full bg-faint shrink-0"
      data-brand={subs[0]}
      title={subs.join(", ")}
    />
  );
}

interface Props {
  agent: string;
  workspace: string;
  surfaces: SurfaceVisibility;
  sessions: SessionInfo[];
  projects: RecentWorkspace[];
  activeSession: string;
  onSwitchAgent: (agent: string) => void;
  onNewSession: (agent: string) => void;
  onSelectSession: (id: string, workspace: string, agent: string) => void;
  onNewProject: (persona: string) => void;
  onRenameSession: (id: string, title: string) => void;
  onDeleteSession: (id: string) => void;
  onArchiveSession: (id: string, archived: boolean) => void;
  onTogglePin: (id: string, pinned: boolean) => void;
  onSetSessionFolder: (id: string, folderId: string | null) => void;
  onApplyMagicSort: (proposals: MagicSortProposal[]) => Promise<MagicSortApplyResult>;
  onArchiveAllSessions: () => void;
  onDeleteFolder: (id: string) => Promise<boolean>;
  onManage: () => void;
  // Grouped-nav gear + New-session menu's "Manage personas…" entry points (§7).
  onOpenPersona: (id: string) => void;
  onManagePersonas: () => void;
  onOpenScheduled: () => void;
  onOpenProjects: (projectId?: string) => void;
  // Scheduled-band row click: open the Automations surface ON that automation (UX-023).
  onOpenAutomation: (id: string) => void;
  onOpenIntegrations: () => void;
  onOpenAudit: () => void;
  onOpenInbox: () => void;
  scheduledActive: boolean;
  projectsActive: boolean;
  integrationsActive: boolean;
  auditActive: boolean;
  inboxActive: boolean;
  // Collapse controls (⌘B / hover-peek). `onCollapse` docks/undocks; `onPeekLeave` hides the
  // floating peek when the pointer leaves the panel.
  collapsed?: boolean;
  onCollapse?: () => void;
  onPeekLeave?: () => void;
}

const UNFILED_KEY = "__unfiled__";

// Compact age for project session rows: "now" / "5m" / "6h" / "3d" / "2w" / "4mo" / "2y".
const compactAge = (iso?: string | null): string => {
  if (!iso) return "";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "";
  const secs = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (secs < 60) return "now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d`;
  const weeks = Math.floor(days / 7);
  if (days < 30) return `${weeks}w`;
  const months = Math.floor(days / 30);
  if (days < 365) return `${months}mo`;
  return `${Math.floor(days / 365)}y`;
};

// Sessions shown per group before "Show more" comes from Settings (sessions_peek, default 5).

export function Sidebar(props: Props) {
  const [searchModalOpen, setSearchModalOpen] = useState(false);
  const [appMenuOpen, setAppMenuOpen] = useState(false);
  // The account row (§26): cloud sign-in status drives the avatar/name/dot; refreshed on
  // focus and whenever the menu opens (sign-in completes out-of-band in the browser).
  const [cloud, setCloud] = useState<CloudStatus | null>(null);
  // Inbox chip sticky unlock (§26): absent until the product first parks an item (or a
  // session first goes Unattended), then permanent. Per-device, like nav collapse.
  const [inboxUnlocked, setInboxUnlocked] = useState(
    () => localStorage.getItem("ocw:inbox-unlocked") === "1",
  );
  const refreshCloud = () => getCloudStatus().then(setCloud).catch(() => {});
  useEffect(() => {
    refreshCloud();
    const onFocus = () => refreshCloud();
    window.addEventListener("focus", onFocus);
    window.addEventListener(CLOUD_CHANGED, onFocus);
    const unlock = () => {
      localStorage.setItem("ocw:inbox-unlocked", "1");
      setInboxUnlocked(true);
    };
    window.addEventListener(INBOX_UNLOCK, unlock);
    return () => {
      window.removeEventListener("focus", onFocus);
      window.removeEventListener(CLOUD_CHANGED, onFocus);
      window.removeEventListener(INBOX_UNLOCK, unlock);
    };
  }, []);
  // UX-023: automations feed the nav row's badge + the Scheduled band. The 15s poll
  // is the baseline; mutations announce AUTOMATIONS_CHANGED for an instant refresh
  // (mark-seen must clear the badge the moment the detail opens).
  const [automations, setAutomations] = useState<Automation[]>([]);
  useEffect(() => {
    const load = () => getAutomations().then(setAutomations).catch(() => {});
    load();
    const t = setInterval(load, 15_000);
    window.addEventListener(AUTOMATIONS_CHANGED, load);
    return () => {
      clearInterval(t);
      window.removeEventListener(AUTOMATIONS_CHANGED, load);
    };
  }, []);
  const [topicProjects, setTopicProjects] = useState<ProjectSummary[]>([]);
  const [collapsedTopicProjects, setCollapsedTopicProjects] = useState<Set<string>>(new Set());
  useEffect(() => {
    const load = async () => {
      // The list route carries session_ids, so one request covers the whole band. Fanning
      // out to getProject() per project re-reads each project's markdown and session list.
      try {
        setTopicProjects(await getProjects());
      } catch {
        setTopicProjects([]);
      }
    };
    load();
    window.addEventListener(PROJECTS_CHANGED, load);
    return () => window.removeEventListener(PROJECTS_CHANGED, load);
  }, []);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [editingFolderId, setEditingFolderId] = useState<string | null>(null);
  const [folderEditValue, setFolderEditValue] = useState("");
  // undefined = idle, null = the folder-layout row, string = a session's folder picker.
  const [creatingFolderForSession, setCreatingFolderForSession] = useState<
    string | null | undefined
  >(undefined);
  // Two-step delete inside the row's ⋮ menu: Delete arms ("Delete?"), a second click deletes.
  // Archive is the primary way to put a conversation away — one click, reversible.
  const [confirmDelId, setConfirmDelId] = useState<string | null>(null);
  const [confirmSweep, setConfirmSweep] = useState(false);
  const [folders, setFolders] = useState<ChatFolder[]>([]);
  // Unfiled is not a stored folder, so it borrows collapsedFolders under a sentinel key
  // rather than carrying its own boolean. Folder ids are uuids, so no collision.
  const [collapsedFolders, setCollapsedFolders] = useState<Set<string>>(new Set());
  const [folderMenu, setFolderMenu] = useState<{
    id: string;
    top: number;
    left: number;
  } | null>(null);
  const [moveMenu, setMoveMenu] = useState<{
    sessionId: string;
    top: number;
    left: number;
  } | null>(null);
  useEffect(() => {
    listFolders().then(setFolders).catch(() => setFolders([]));
  }, []);
  // The open row-actions ⋮ menu (one at a time). Fixed-position, not absolute: the expanded
  // accordion group clips overflow (its rounded fill), so an absolute popover on its lower rows
  // would be cut off — same constraint as SlackDetail's person picker.
  const [rowMenu, setRowMenu] = useState<{
    id: string;
    top: number;
    left: number;
    anchor: HTMLElement;
  } | null>(null);
  const closeRowMenu = () => {
    setRowMenu(null);
    setMoveMenu(null);
    setConfirmDelId(null);
  };
  const openRowMenu = (id: string, anchor: HTMLElement) => {
    const r = anchor.getBoundingClientRect();
    const MENU_W = 160; // w-40
    const MENU_H = 190; // five items + divider; only used to flip upward near the window bottom
    setConfirmDelId(null);
    setRowMenu({
      id,
      top: r.bottom + 4 + MENU_H > window.innerHeight ? r.top - MENU_H : r.bottom + 4,
      left: Math.max(8, r.right - MENU_W),
      anchor,
    });
  };
  useEffect(() => {
    if (!rowMenu) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && closeRowMenu();
    // Scrolling an ANCESTOR of the anchor row detaches the fixed menu from it — dismiss.
    // Filter by containment: unrelated scrollers (the transcript auto-follow during a
    // streaming turn fires constantly) must not close the menu.
    const onScroll = (e: Event) => {
      const t = e.target;
      if (t === document || (t instanceof Node && t.contains(rowMenu.anchor))) closeRowMenu();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("scroll", onScroll, true);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onScroll, true);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rowMenu]);
  const [showArchived, setShowArchived] = useState(false);
  // Surfaced + enabled personas drive the surface list + family-aware behavior.
  // Refetched on the personas-changed event so an enable/install/delete in Settings
  // shows up here immediately (no page refresh).
  const [personas, setPersonas] = useState<Persona[] | null>(null);
  useEffect(() => {
    const load = () =>
      getPersonas()
        .then(setPersonas)
        .catch(() => setPersonas(null));
    load();
    window.addEventListener(PERSONAS_CHANGED, load);
    return () => window.removeEventListener(PERSONAS_CHANGED, load);
  }, []);
  const personaOf = (id: string) => personas?.find((p) => p.id === id);

  // Sidebar layout (§7): "grouped" = the per-persona accordion; "flat" = a single ungrouped list
  // (Pinned + Recent). Read the persisted preference on load; ABSENT falls back by the
  // Personas flag — with personas hidden for launch, a per-persona accordion groups by
  // a concept the user can't see, so the default is the flat chronological list
  // (owner call 2026-07-20). An explicit stored choice always wins.
  const defaultLayout: "flat" | "grouped" | "folder" = showPersonas() ? "grouped" : "flat";
  const [layout, setLayout] = useState<"flat" | "grouped" | "folder">(defaultLayout);
  // Sessions shown per group before "Show more" — Settings ▸ Appearance ▸ Sidebar.
  const [peek, setPeek] = useState(5);
  useEffect(() => {
    getSettings()
      .then((s) => {
        setLayout(
          s.nav_layout === "flat"
            ? "flat"
            : s.nav_layout === "grouped"
              ? "grouped"
              : s.nav_layout === "folder"
                ? "folder"
                : defaultLayout,
        );
        if (s.sessions_peek) setPeek(s.sessions_peek);
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const setGroupBy = (next: "flat" | "grouped" | "folder") => {
    setLayout(next);
    setNavLayout(next).catch(() => {});
  };
  // Chronological RECENT list: cap at RECENT_PEEK with a Show more/less toggle so the sidebar
  // doesn't grow unbounded.
  const RECENT_PEEK = 4;
  const [recentExpanded, setRecentExpanded] = useState(false);
  // The RECENT-header group/filter popover (§20). Filter = show only these personas (empty = all).
  const [groupMenuOpen, setGroupMenuOpen] = useState(false);
  const [magicSortState, setMagicSortState] = useState<
    "idle" | "loading" | "preview" | "empty" | "error" | "applying"
  >("idle");
  const [magicSortProposals, setMagicSortProposals] = useState<MagicSortProposal[]>([]);
  const [magicSortExcluded, setMagicSortExcluded] = useState<Set<string>>(new Set());
  const [magicSortCounts, setMagicSortCounts] = useState({ considered: 0, skipped: 0 });
  const magicSortRequest = useRef(0);
  const [filterPersonas, setFilterPersonas] = useState<Set<string>>(new Set());
  const toggleFilterPersona = (id: string) =>
    setFilterPersonas((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  const personaVisible = (agent: string) =>
    filterPersonas.size === 0 || filterPersonas.has(agent);

  const startMagicSort = async () => {
    const request = magicSortRequest.current + 1;
    magicSortRequest.current = request;
    setGroupMenuOpen(false);
    setMagicSortState("loading");
    setMagicSortProposals([]);
    setMagicSortExcluded(new Set());
    try {
      const result = await proposeMagicSort();
      if (magicSortRequest.current !== request) return;
      if (!result.ok) {
        setMagicSortState("error");
        return;
      }
      const proposals = result.proposals || [];
      setMagicSortCounts({
        considered: result.considered || 0,
        skipped: result.skipped || 0,
      });
      setMagicSortProposals(proposals);
      setMagicSortState(proposals.length ? "preview" : "empty");
    } catch {
      if (magicSortRequest.current !== request) return;
      setMagicSortState("error");
    }
  };

  const cancelMagicSort = () => {
    magicSortRequest.current += 1;
    setMagicSortState("idle");
    setMagicSortProposals([]);
    setMagicSortExcluded(new Set());
  };

  const submitMagicSort = async () => {
    const selected = magicSortProposals.filter(
      (proposal) => !magicSortExcluded.has(proposal.session_id),
    );
    if (!selected.length) return;
    setMagicSortState("applying");
    try {
      const result = await props.onApplyMagicSort(selected);
      if (!result.ok) {
        setMagicSortState("error");
        return;
      }
      listFolders().then(setFolders).catch(() => {});
      cancelMagicSort();
    } catch {
      setMagicSortState("error");
    }
  };

  // Which accordion body is expanded. Decoupled from the active session (props.agent): expanding
  // a persona BROWSES its sessions without switching the chat area. Selecting a session or "New
  // session" is what switches (and re-opens that persona). Falls back to the active persona.
  const [openKey, setOpenKey] = useState<string | null>(props.agent);
  useEffect(() => setOpenKey(props.agent), [props.agent]);
  const browseKey = openKey ?? props.agent; // the persona whose sessions the body shows

  // Per-project collapse + "Show more". The active workspace's folder is open by default; toggling
  // any folder flips it (XOR). `projShowAll` lifts the peek cap for a given folder;
  // `personaShowAll` does the same for a (non-project) persona's flat session list.
  const [projToggled, setProjToggled] = useState<Set<string>>(new Set());
  const [projShowAll, setProjShowAll] = useState<Set<string>>(new Set());
  const [personaShowAll, setPersonaShowAll] = useState<Set<string>>(new Set());
  const toggleSet = (set: Set<string>, key: string) => {
    const next = new Set(set);
    next.has(key) ? next.delete(key) : next.add(key);
    return next;
  };

  // Pinned sessions across ALL personas — the cross-persona band at the top (manual pins only).
  const pinnedSessions = props.sessions.filter(
    (s) => s.pinned && !s.session_id.startsWith("__") && !s.archived,
  );
  // §31 (revised 2026-07-21): mention-spawned sessions list chronologically in Recent like any
  // other session — the OriginIcon in the row's indicator cluster marks where they came from.
  // The separate collapsed "From Slack" band hid fresh mentions below week-old sessions.
  // A row in the account menu (§26): closes the menu, then runs the destination.
  const appMenuItem = (
    icon: IconName,
    label: string,
    onClick: () => void,
    active?: boolean,
    trailing?: ReactNode,
  ) => (
    <button
      className={
        "w-full flex items-center gap-2.5 px-3 py-1.5 text-[13px] text-left " +
        (active ? "text-ink bg-paper" : "hover:bg-paper")
      }
      onClick={() => {
        setAppMenuOpen(false);
        onClick();
      }}
    >
      <Icon name={icon} size={15} className="shrink-0 text-muted" />
      <span className="flex-1">{label}</span>
      {/* aria-hidden: the badge/shortcut must not leak into the accessible name (the old
          Inbox row's name-includes-the-badge-count nuisance, not repeated). */}
      {trailing != null && <span aria-hidden>{trailing}</span>}
    </button>
  );

  // Display identity for the account row: the cloud profile only carries the email, so the
  // row shows the capitalized local part ("rohit@…" → "Rohit"); the menu header shows it all.
  const accountEmail = cloud?.signed_in ? cloud.account : "";
  const accountName = accountEmail
    ? accountEmail.split("@")[0].replace(/^./, (c) => c.toUpperCase())
    : "";

  // Roll the per-session attention/liveness up to the persona header and the footer Inbox: the
  // accent count bubbles (sum), the liveness dot aggregates (working wins over sleeping).
  const attnByPersona = new Map<string, number>();
  const liveByPersona = new Map<string, "working" | "sleeping">();
  let totalAttention = 0;
  for (const s of props.sessions) {
    if (s.session_id.startsWith("__") || s.archived) continue;
    const a = s.attention || 0;
    if (a > 0) {
      attnByPersona.set(s.agent, (attnByPersona.get(s.agent) || 0) + a);
      totalAttention += a;
    }
    if (s.liveness === "working") liveByPersona.set(s.agent, "working");
    else if (s.liveness === "sleeping" && liveByPersona.get(s.agent) !== "working")
      liveByPersona.set(s.agent, "sleeping");
  }

  // First pending item ever observed → the inbox chip unlocks and stays (§26 sticky unlock).
  useEffect(() => {
    if (totalAttention > 0 && !inboxUnlocked) {
      localStorage.setItem("ocw:inbox-unlocked", "1");
      setInboxUnlocked(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [totalAttention]);

  // Body data is keyed to the BROWSED persona (only one body renders at a time). Pinned sessions are
  // EXCLUDED here: they live in the cross-persona Pinned band only, so they don't repeat inside the
  // persona group / project list (matching the flat layout's Recent, which also drops pinned).
  const all = props.sessions.filter((s) => s.agent === browseKey && !s.session_id.startsWith("__"));
  const mine = all.filter((s) => !s.archived && !s.pinned);
  const allArchived = props.sessions.filter(
    (s) => s.archived && !s.session_id.startsWith("__") && personaVisible(s.agent),
  );
  // Only PROJECT-SCOPED personas group sessions by project (git-bound Code, project-bound Ops).
  // Scratch/deliverable conversations are orphan (each has its own per-conversation scratch dir),
  // so they list flat. Workspace-aware (not id-aware) — any git/project persona gets Projects.
  const workspaceSurface = isProjectScoped(personaOf(browseKey));

  // Search now lives in the SearchModal (command-palette overlay), so the sidebar lists never filter
  // in place — these stay constant and the `.filter(matches)` / `normalizedQuery ? …` call sites
  // below are intentional no-ops kept to avoid churn.
  const normalizedQuery = "";
  const matches = (_s: SessionInfo) => true;

  // Recent = every non-pinned, non-archived, real session across ALL personas, newest first
  // (by updated_at; missing timestamps keep store order), search-filtered. Drives the flat layout.
  const recentSessions = [...props.sessions]
    .filter((s) => !s.archived && !s.session_id.startsWith("__") && !s.pinned)
    .filter((s) => personaVisible(s.agent))
    .filter(matches)
    .sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));

  const cancelFolderEdit = () => {
    setEditingFolderId(null);
    setCreatingFolderForSession(undefined);
    setFolderEditValue("");
  };
  const startCreatingFolder = (sessionId: string | null) => {
    setEditingFolderId(null);
    setCreatingFolderForSession(sessionId);
    setFolderEditValue("");
  };
  const createChatFolder = async (name: string, sessionId?: string) => {
    const next = name.trim();
    if (!next) {
      cancelFolderEdit();
      return;
    }
    const result = await createFolder(next);
    if (!result.ok || !result.folder) return;
    setFolders((current) => [...current, result.folder!]);
    setCollapsedFolders((current) => {
      const next = new Set(current);
      next.delete(result.folder!.id);
      return next;
    });
    if (sessionId) props.onSetSessionFolder(sessionId, result.folder.id);
    cancelFolderEdit();
    if (sessionId) closeRowMenu();
  };
  const renameChatFolder = async (folder: ChatFolder) => {
    const next = folderEditValue.trim();
    if (!next || next === folder.name) {
      cancelFolderEdit();
      return;
    }
    const result = await renameFolder(folder.id, next);
    if (!result.ok || !result.folder) return;
    setFolders((current) =>
      current.map((entry) => (entry.id === folder.id ? result.folder! : entry)),
    );
    cancelFolderEdit();
  };

  const knownFolderIds = new Set(folders.map((folder) => folder.id));
  const sessionsForFolder = (folderId: string) =>
    recentSessions.filter((session) => session.folder_id === folderId);
  const unfiledSessions = recentSessions.filter(
    (session) => !session.folder_id || !knownFolderIds.has(session.folder_id),
  );
  const folderAttention = (sessions: SessionInfo[]) =>
    sessions.reduce((total, session) => total + (session.attention || 0), 0);
  const folderLiveness = (sessions: SessionInfo[]): "working" | "sleeping" | undefined =>
    sessions.some((session) => session.liveness === "working")
      ? "working"
      : sessions.some((session) => session.liveness === "sleeping")
        ? "sleeping"
        : undefined;

  const folderActions = (folder: ChatFolder) => {
    const menuOpen = folderMenu?.id === folder.id;
    const openMenu = (anchor: HTMLElement) => {
      const rect = anchor.getBoundingClientRect();
      setFolderMenu({
        id: folder.id,
        top: rect.bottom + 72 > window.innerHeight ? rect.top - 72 : rect.bottom + 4,
        left: Math.max(8, rect.right - 144),
      });
    };
    const item = (icon: IconName, label: string, onClick: () => void) => (
      <button
        className="w-full flex items-center gap-2 px-2.5 py-1.5 text-[12.5px] text-left hover:bg-paper active:bg-line/60"
        role="menuitem"
        onClick={() => {
          setFolderMenu(null);
          onClick();
        }}
      >
        <Icon name={icon} size={13} className="shrink-0 text-muted" />
        <span>{label}</span>
      </button>
    );
    return (
      <span className="shrink-0" onClick={(event) => event.stopPropagation()}>
        <button
          className={
            "w-6 h-6 grid place-items-center rounded-md hover:bg-panel active:bg-line/60 " +
            (menuOpen ? "text-ink bg-panel" : "text-faint hover:text-ink")
          }
          aria-label={`Folder actions for ${folder.name}`}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          onClick={(event) => (menuOpen ? setFolderMenu(null) : openMenu(event.currentTarget))}
        >
          <Icon name="moreHorizontal" size={14} className="rotate-90" />
        </button>
        {menuOpen && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setFolderMenu(null)} />
            <div
              className="fixed z-50 w-36 rounded-xl border border-line bg-panel shadow-xl py-1"
              style={{ top: folderMenu.top, left: folderMenu.left }}
              role="menu"
            >
              {item("pencil", "Rename", () => {
                setCreatingFolderForSession(undefined);
                setEditingFolderId(folder.id);
                setFolderEditValue(folder.name);
              })}
              {item("trash", "Delete", async () => {
                if (await props.onDeleteFolder(folder.id)) {
                  setFolders((current) => current.filter((entry) => entry.id !== folder.id));
                }
              })}
            </div>
          </>
        )}
      </span>
    );
  };

  // Row actions live behind ONE ⋮ kebab per row (FB-011: four hover icons read as clutter) —
  // the menu offers Rename · Pin/Unpin · Archive/Unarchive · Delete, with the two-step delete
  // confirm kept inside it. Shared by BOTH row styles, so the chronological cardRow offers the
  // same actions as the persona accordion's sessionRow (owner ask 2026-07-09).
  const rowActions = (s: SessionInfo, title: string) => {
    const menuOpen = rowMenu?.id === s.session_id;
    const item = (testid: string, icon: IconName, label: string, onClick: () => void) => (
      <button
        className="w-full flex items-center gap-2 px-2.5 py-1.5 text-[12.5px] text-left hover:bg-paper"
        data-testid={testid}
        role="menuitem"
        onClick={() => {
          closeRowMenu();
          onClick();
        }}
      >
        <Icon name={icon} size={13} className="shrink-0 text-muted" />
        <span className="flex-1">{label}</span>
      </button>
    );
    return (
      <span
        // Stay visible while this row's menu is open — the pointer may be on the menu, off the row.
        className={(menuOpen ? "flex" : "hidden group-hover:flex") + " items-center shrink-0"}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          title="Session actions"
          aria-label="Session actions"
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          data-testid="row-menu"
          className={
            "w-5 h-5 grid place-items-center rounded hover:bg-paper " +
            (menuOpen ? "text-ink bg-paper" : "text-faint hover:text-ink")
          }
          onClick={(e) => (menuOpen ? closeRowMenu() : openRowMenu(s.session_id, e.currentTarget))}
        >
          {/* Vertical kebab = the horizontal glyph rotated — no extra icon needed. */}
          <Icon name="moreHorizontal" size={14} className="rotate-90" />
        </button>
        {menuOpen && (
          <>
            <div className="fixed inset-0 z-40" onClick={closeRowMenu} />
            <div
              className="fixed z-50 w-40 rounded-xl border border-line bg-panel shadow-xl py-1"
              style={{ top: rowMenu!.top, left: rowMenu!.left }}
              role="menu"
            >
              {item("row-menu-rename", "pencil", "Rename", () => {
                setEditingId(s.session_id);
                setEditValue(title);
              })}
              {item("row-menu-pin", "pin", s.pinned ? "Unpin" : "Pin", () =>
                props.onTogglePin(s.session_id, !s.pinned),
              )}
              <button
                className="w-full flex items-center gap-2 px-2.5 py-1.5 text-[12.5px] text-left hover:bg-paper active:bg-line/60"
                data-testid="row-menu-folder"
                role="menuitem"
                aria-haspopup="menu"
                onClick={(event) => {
                  const rect = event.currentTarget.getBoundingClientRect();
                  const width = 176;
                  setMoveMenu({
                    sessionId: s.session_id,
                    top: Math.max(8, Math.min(rect.top, window.innerHeight - 248)),
                    left:
                      rect.right + width <= window.innerWidth
                        ? rect.right + 4
                        : Math.max(8, rect.left - width - 4),
                  });
                }}
              >
                <Icon name="folder" size={13} className="shrink-0 text-muted" />
                <span className="flex-1">Move to folder…</span>
                <Icon name="chevronRight" size={12} className="text-faint" />
              </button>
              {item("row-menu-archive", "archive", s.archived ? "Unarchive" : "Archive", () =>
                props.onArchiveSession(s.session_id, !s.archived),
              )}
              <div className="h-px bg-line my-1 mx-2" />
              {confirmDelId === s.session_id ? (
                <button
                  title="Click again to permanently delete"
                  className="w-full flex items-center gap-2 px-2.5 py-1.5 text-[12.5px] text-left font-medium text-danger hover:bg-paper"
                  data-testid="row-menu-delete"
                  role="menuitem"
                  onClick={() => {
                    closeRowMenu();
                    props.onDeleteSession(s.session_id);
                  }}
                >
                  <Icon name="trash" size={13} className="shrink-0" />
                  <span className="flex-1">Delete?</span>
                </button>
              ) : (
                <button
                  className="w-full flex items-center gap-2 px-2.5 py-1.5 text-[12.5px] text-left text-danger hover:bg-paper"
                  data-testid="row-menu-delete"
                  role="menuitem"
                  onClick={() => setConfirmDelId(s.session_id)}
                >
                  <Icon name="trash" size={13} className="shrink-0" />
                  <span className="flex-1">Delete</span>
                </button>
              )}
            </div>
            {moveMenu?.sessionId === s.session_id && (
              <div
                className="fixed z-[60] w-44 max-h-64 overflow-y-auto rounded-xl border border-line bg-panel shadow-xl py-1"
                style={{ top: moveMenu.top, left: moveMenu.left }}
                role="menu"
                data-testid="folder-picker-menu"
              >
                <button
                  className="w-full px-2.5 py-1.5 text-[12.5px] text-left hover:bg-paper active:bg-line/60"
                  role="menuitem"
                  onClick={() => {
                    closeRowMenu();
                    props.onSetSessionFolder(s.session_id, null);
                  }}
                >
                  Unfiled
                </button>
                {folders.map((folder) => (
                  <button
                    key={folder.id}
                    className="w-full px-2.5 py-1.5 text-[12.5px] text-left truncate hover:bg-paper active:bg-line/60"
                    role="menuitem"
                    onClick={() => {
                      closeRowMenu();
                      props.onSetSessionFolder(s.session_id, folder.id);
                    }}
                  >
                    {folder.name}
                  </button>
                ))}
                <div className="h-px bg-line my-1 mx-2" />
                {creatingFolderForSession === s.session_id ? (
                  <div className="w-full flex items-center px-2.5 py-1">
                    <input
                      className="flex-1 min-w-0 px-1.5 py-0.5 rounded-md bg-panel border border-accent text-[13px] text-ink outline-none"
                      value={folderEditValue}
                      placeholder="Folder name"
                      autoFocus
                      onFocus={(event) => event.currentTarget.select()}
                      onClick={(event) => event.stopPropagation()}
                      onChange={(event) => setFolderEditValue(event.target.value)}
                      onBlur={() => void createChatFolder(folderEditValue, s.session_id)}
                      onKeyDown={(event) => {
                        event.stopPropagation();
                        if (event.key === "Enter") {
                          void createChatFolder(folderEditValue, s.session_id);
                        } else if (event.key === "Escape") {
                          cancelFolderEdit();
                        }
                      }}
                    />
                  </div>
                ) : (
                  <button
                    className="w-full flex items-center gap-2 px-2.5 py-1.5 text-[12.5px] text-left text-accent hover:bg-paper active:bg-line/60"
                    role="menuitem"
                    onClick={() => startCreatingFolder(s.session_id)}
                  >
                    <Icon name="folderPlus" size={13} />
                    <span>New folder…</span>
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </span>
    );
  };

  // A compact session row (mock §141 grouped/recent rows): one-line title + right-side indicators,
  // with the ⋮ actions kebab revealed on hover. Used in accordion bodies + grouped cards.
  const sessionRow = (s: SessionInfo, opts: { showTime?: boolean } = {}) => {
    const title = s.title || s.session_id;
    const editing = editingId === s.session_id;
    const active = s.session_id === props.activeSession;
    const commitRename = () => {
      const next = editValue.trim();
      if (next && next !== title) props.onRenameSession(s.session_id, next);
      setEditingId(null);
    };
    return (
      <div
        key={s.session_id}
        className={
          "group flex items-center gap-2 px-2 py-1.5 rounded-lg text-left cursor-pointer " +
          (active
            ? "bg-ink/[0.055]"
            : "hover:bg-panel")
        }
        onClick={() => {
          if (!editing) props.onSelectSession(s.session_id, s.workspace, s.agent);
        }}
        title={editing ? undefined : title}
      >
        {editing ? (
          <input
            className="flex-1 min-w-0 px-1.5 py-0.5 rounded-md bg-panel border border-accent text-[13px] text-ink outline-none"
            value={editValue}
            autoFocus
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === "Enter") commitRename();
              else if (e.key === "Escape") setEditingId(null);
            }}
          />
        ) : (
          <>
            <span
              className={
                "min-w-0 flex-1 flex items-center gap-1.5 truncate text-[13px] " +
                (active ? "font-medium text-ink" : "text-ink")
              }
            >
              {s.pinned && <Icon name="pin" size={11} className="text-faint shrink-0" />}
              <span className="truncate">{title}</span>
            </span>
            <span
              className={
                "flex items-center gap-1.5 shrink-0 group-hover:hidden" +
                (rowMenu?.id === s.session_id ? " hidden" : "")
              }
            >
              {opts.showTime && compactAge(s.updated_at) && (
                <span className="text-[11px] text-faint tabular-nums">{compactAge(s.updated_at)}</span>
              )}
              <OriginIcon s={s} />
              <LiveDot state={s.liveness} />
              <AttnBadge n={s.attention || 0} />
            </span>
            {rowActions(s, title)}
          </>
        )}
      </div>
    );
  };

  // A single-line card row (mock §141 list-flat, subtitle dropped 2026-07-21): title +
  // right-side indicators, with the ⋮ actions kebab revealed on hover. Shared by the flat
  // layout's Pinned and Recent sections. Personas are disabled for the first release; when
  // they return, surface the persona on hover (e.g. in the row tooltip) — not as a subtitle.
  const cardRow = (s: SessionInfo) => {
    const active = s.session_id === props.activeSession;
    const title = s.title || s.session_id;
    const editing = editingId === s.session_id;
    const commitRename = () => {
      const next = editValue.trim();
      if (next && next !== title) props.onRenameSession(s.session_id, next);
      setEditingId(null);
    };
    return (
      <div
        key={s.session_id}
        className={
          "group w-full flex items-center gap-2.5 px-2 py-2 rounded-lg cursor-pointer text-left " +
          (active
            ? "bg-ink/[0.055]"
            : "hover:bg-paper")
        }
        title={editing ? undefined : title}
        onClick={() => {
          if (!editing) props.onSelectSession(s.session_id, s.workspace, s.agent);
        }}
      >
        {/* No leading glyph on session rows (Rohit's call 2026-07-07: the per-session icon
            read as noise in both grouped and chronological). */}
        {editing ? (
          <input
            className="flex-1 min-w-0 px-1.5 py-0.5 rounded-md bg-panel border border-accent text-[13px] text-ink outline-none"
            value={editValue}
            autoFocus
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === "Enter") commitRename();
              else if (e.key === "Escape") setEditingId(null);
            }}
          />
        ) : (
          <>
            <span className="min-w-0 flex-1 block truncate text-[13px] font-medium">
              {title}
            </span>
            <span
              className={
                "flex items-center gap-1.5 shrink-0 group-hover:hidden" +
                (rowMenu?.id === s.session_id ? " hidden" : "")
              }
            >
              <OriginIcon s={s} />
              <ConnectorDot subs={s.subscriptions} />
              <LiveDot state={s.liveness} />
              <AttnBadge n={s.attention || 0} />
            </span>
            {rowActions(s, title)}
          </>
        )}
      </div>
    );
  };

  // The cross-persona Pinned band (manual pins only) — icon-free rows. Appears in BOTH layouts
  // (flat list AND accordion), so it's factored here for reuse.
  const pinnedBand = () =>
    pinnedSessions.length > 0 ? (
      <div>
        <div className="px-1.5 text-[10.5px] uppercase tracking-[0.07em] text-faint font-semibold mb-1">
          Pinned
        </div>
        <div className="space-y-0.5">
          {pinnedSessions.map((s) => cardRow(s))}
        </div>
      </div>
    ) : null;

  // UX-023: the Scheduled band — ONE entry per automation (never per run): name +
  // cadence, with the unseen-runs badge. Runs themselves never enter Recent (run
  // sessions are __run__-prefixed and hidden from the sessions list).
  const scheduledBand = () =>
    automations.length > 0 ? (
      <div data-testid="scheduled-band">
        <div className="px-1.5 text-[10.5px] uppercase tracking-[0.07em] text-faint font-semibold mb-1">
          Scheduled
        </div>
        <div className="space-y-0.5">
          {automations.map((a) => (
            <button
              key={a.id}
              className="w-full flex items-center gap-2 px-1.5 py-1 rounded-lg text-left hover:bg-paper"
              data-testid={`scheduled-${a.id}`}
              title={a.title}
              onClick={() => props.onOpenAutomation(a.id)}
            >
              <div className="flex-1 min-w-0">
                <div className="text-[13px] text-ink truncate">{a.title}</div>
                <div className="text-[11px] text-faint truncate">{a.schedule}</div>
              </div>
              <UnseenBadge n={a.unseen_runs || 0} failed={a.unseen_failed} />
            </button>
          ))}
        </div>
      </div>
    ) : null;

  const projectsBand = () =>
    topicProjects.length > 0 ? (
      <div data-testid="projects-band">
        <div className="px-1.5 text-[10.5px] uppercase tracking-[0.07em] text-faint font-semibold mb-1">
          Projects
        </div>
        <div className="space-y-1.5">
          {topicProjects.map((project) => {
            const expanded = !collapsedTopicProjects.has(project.id);
            const knownSessionIds = new Set(project.session_ids);
            const projectSessions = props.sessions.filter(
              (session) =>
                !session.archived &&
                (session.project_id === project.id || knownSessionIds.has(session.session_id)),
            );
            return (
              <div
                key={project.id}
                className={expanded ? "rounded-xl bg-paper/70 overflow-hidden" : ""}
                data-testid={`project-folder-${project.id}`}
              >
                <div className="flex items-center px-2 py-1">
                  <button
                    className="min-w-0 flex-1 flex items-center gap-2 py-1 text-left rounded-lg hover:text-accent active:opacity-70"
                    onClick={() => props.onOpenProjects(project.id)}
                    title={`Open project ${project.name}`}
                  >
                    <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">
                      {project.name}
                    </span>
                    <LiveDot state={folderLiveness(projectSessions)} />
                    <AttnBadge n={folderAttention(projectSessions)} />
                  </button>
                  <button
                    className="w-10 h-10 -my-1 grid place-items-center rounded-lg text-faint hover:text-ink hover:bg-panel active:bg-line/60"
                    aria-label={`${expanded ? "Collapse" : "Expand"} ${project.name}`}
                    onClick={() =>
                      setCollapsedTopicProjects((current) => toggleSet(current, project.id))
                    }
                  >
                    <Icon name={expanded ? "chevronDown" : "chevronRight"} size={15} />
                  </button>
                </div>
                {expanded && (
                  <div className="px-1.5 pb-1.5 space-y-0.5">
                    {projectSessions.length > 0 ? (
                      projectSessions.map((session) => sessionRow(session, { showTime: true }))
                    ) : (
                      <div className="px-2 py-1.5 text-[12px] text-faint">No conversations</div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    ) : null;

  const archiveFolder = () => {
    if (allArchived.length === 0) return null;
    // Archive obeys the search box like every other band — without this a query returns
    // its matches plus the entire archive, which reads as the search being broken.
    const shown = allArchived.filter(matches);
    if (normalizedQuery && shown.length === 0) return null;
    return (
      <div
        className={showArchived ? "rounded-xl bg-paper/70 overflow-hidden" : ""}
        data-testid="archive-folder"
      >
        <button
          className={
            "w-full flex items-center gap-2 px-2 py-2 text-left select-none " +
            (showArchived ? "" : "rounded-lg hover:bg-paper active:bg-line/60")
          }
          onClick={() => setShowArchived((value) => !value)}
        >
          <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">
            Archive
          </span>
          <span className="text-[11px] text-faint tabular-nums">{shown.length}</span>
          <Icon
            name={showArchived ? "chevronDown" : "chevronRight"}
            size={15}
            className="text-faint shrink-0"
          />
        </button>
        {showArchived && (
          <div className="px-1.5 pb-1.5 space-y-0.5">
            {shown.map((session) => sessionRow(session, { showTime: true }))}
          </div>
        )}
      </div>
    );
  };

  const magicSortPanel = () => {
    if (magicSortState === "idle") return null;
    const selectedCount = magicSortProposals.length - magicSortExcluded.size;
    const isReview = magicSortState === "preview" || magicSortState === "applying";
    return (
      <div
        className="absolute right-0 top-7 z-50 w-[min(320px,calc(100vw-24px))] rounded-xl border border-line bg-panel shadow-xl overflow-hidden"
        data-testid="magic-sort-panel"
      >
        <div className="flex items-center gap-2 px-3 py-2.5 border-b border-line">
          <Icon
            name={magicSortState === "loading" ? "refresh" : "sparkle"}
            size={14}
            className={magicSortState === "loading" ? "animate-spin motion-reduce:animate-none text-accent" : "text-accent"}
          />
          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-medium text-ink">
              {magicSortState === "loading"
                ? "Sorting…"
                : isReview
                  ? "Review Magic sort"
                  : magicSortState === "empty"
                    ? "Nothing to sort"
                    : "Could not sort right now"}
            </div>
            {isReview && (
              <div className="text-[11px] text-faint tabular-nums">
                {magicSortCounts.considered} considered
                {magicSortCounts.skipped ? ` · ${magicSortCounts.skipped} skipped` : ""}
              </div>
            )}
          </div>
        </div>

        {magicSortState === "loading" && (
          <div className="px-3 py-5 text-[12px] text-muted">Finding obvious homes for recent chats.</div>
        )}
        {magicSortState === "empty" && (
          <div className="px-3 py-4 text-[12px] text-muted">Your recent chats already look tidy.</div>
        )}
        {magicSortState === "error" && (
          <div className="px-3 py-4 text-[12px] text-muted">Try again in a moment.</div>
        )}
        {isReview && (
          <div className="max-h-72 overflow-y-auto overscroll-contain p-1.5" data-testid="magic-sort-list">
            {magicSortProposals.map((proposal) => {
              const excluded = magicSortExcluded.has(proposal.session_id);
              return (
                <div
                  key={proposal.session_id}
                  className={
                    "flex items-center gap-1 rounded-lg px-2 py-1 transition-colors " +
                    (excluded ? "opacity-45" : "hover:bg-paper")
                  }
                >
                  <div className="min-w-0 flex-1 py-1">
                    <div className="text-[12px] text-ink truncate" title={proposal.title}>
                      {proposal.title}
                    </div>
                    <div className="flex items-center gap-1 text-[11px] text-faint min-w-0">
                      <span aria-hidden>→</span>
                      <span className="truncate">{proposal.target_name}</span>
                      {proposal.action === "new_folder" && (
                        <span className="shrink-0 rounded bg-accent/10 px-1 text-[9px] font-semibold uppercase tracking-wide text-accent">
                          new
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    className="w-10 h-10 shrink-0 grid place-items-center rounded-lg text-faint hover:text-ink hover:bg-line/50 active:bg-line focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
                    aria-label={excluded ? `Include ${proposal.title}` : `Exclude ${proposal.title}`}
                    aria-pressed={excluded}
                    onClick={() =>
                      setMagicSortExcluded((current) => {
                        const next = new Set(current);
                        excluded ? next.delete(proposal.session_id) : next.add(proposal.session_id);
                        return next;
                      })
                    }
                  >
                    <Icon name={excluded ? "refresh" : "x"} size={14} />
                  </button>
                </div>
              );
            })}
          </div>
        )}

        <div className="flex items-center justify-end gap-2 px-2.5 py-2 border-t border-line">
          <button
            className="min-h-10 px-3 rounded-lg text-[12px] text-muted hover:text-ink hover:bg-paper active:bg-line/60 disabled:opacity-45 disabled:cursor-not-allowed"
            disabled={magicSortState === "applying"}
            onClick={cancelMagicSort}
          >
            {isReview ? "Cancel" : "Close"}
          </button>
          {isReview && (
            <button
              className="min-h-10 px-3 rounded-lg bg-accent text-white text-[12px] font-medium hover:brightness-105 active:brightness-95 disabled:opacity-45 disabled:cursor-not-allowed tabular-nums"
              disabled={magicSortState === "applying" || selectedCount === 0}
              onClick={() => void submitMagicSort()}
            >
              {magicSortState === "applying" ? "Applying…" : `Apply ${selectedCount}`}
            </button>
          )}
        </div>
      </div>
    );
  };

  // RECENT header with the group/filter control (§20) — the group toggle moved off the brand bar.
  // "Group by" flips the persona accordion ↔ chronological list; "Filter by coworker" narrows to
  // the checked personas (none checked = all shown).
  const recentHeader = () => {
    const filterPersonaList = (personas || []).filter(
      (p) => (p.enabled && p.surfaced) || agentsWithSessions.has(p.id),
    );
    return (
    <div className="relative flex items-center justify-between px-1.5 mb-1" data-testid="recent-header">
      <span className="text-[10.5px] uppercase tracking-[0.07em] text-faint font-semibold">
        Recent
      </span>
      <div className="flex items-center gap-0.5 -mr-1">
        {/* Armed sweep disarms on any outside click, the same way the row/group menus
            close — otherwise it stays armed indefinitely and a much later click reads
            as "Sweep" but fires the archive. */}
        {confirmSweep && (
          <div className="fixed inset-0 z-40" onClick={() => setConfirmSweep(false)} />
        )}
        <button
          className={
            "h-6 flex items-center gap-1 px-1.5 rounded-md text-[11px] hover:bg-paper active:bg-line/60 " +
            (confirmSweep ? "relative z-50 font-medium text-danger" : "text-faint hover:text-ink")
          }
          title={confirmSweep ? "Click again to archive every conversation" : "Archive all conversations"}
          data-testid="sweep-archive"
          onClick={() => {
            if (!confirmSweep) {
              setConfirmSweep(true);
              return;
            }
            setConfirmSweep(false);
            props.onArchiveAllSessions();
          }}
        >
          <Icon name="archive" size={12} />
          <span>{confirmSweep ? "Archive all?" : "Sweep"}</span>
        </button>
        <button
          className="w-6 h-6 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper active:bg-line/60 disabled:opacity-45 disabled:cursor-not-allowed"
          title="Group & filter conversations"
          aria-label="Group and filter conversations"
          disabled={magicSortState !== "idle"}
          onClick={() => setGroupMenuOpen((v) => !v)}
        >
          <Icon name="sliders" size={14} />
        </button>
      </div>
      {groupMenuOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setGroupMenuOpen(false)} />
          <div
            className="absolute right-0 top-7 z-50 w-56 rounded-xl border border-line bg-panel shadow-xl p-1.5"
            role="menu"
            data-testid="group-filter-menu"
          >
            <div className="px-2 pt-1 pb-1 text-[10.5px] uppercase tracking-[0.06em] text-faint font-semibold">
              Group by
            </div>
            {([["grouped", "Persona"], ["folder", "Folder"], ["flat", "Chronological"]] as [
              "flat" | "grouped" | "folder",
              string,
            ][]).map(
              ([key, label]) => (
                <button
                  key={key}
                  className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-[13px] text-left hover:bg-paper"
                  onClick={() => setGroupBy(key)}
                >
                  <span className="flex-1">{label}</span>
                  {layout === key && <span className="text-accent text-[12px]">✓</span>}
                </button>
              ),
            )}
            <div className="my-1 border-t border-line" />
            <button
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-[13px] text-left text-ink hover:bg-paper active:bg-line/60"
              data-testid="magic-sort-menu-item"
              onClick={() => void startMagicSort()}
            >
              <Icon name="sparkle" size={13} className="text-accent" />
              <span>Magic sort</span>
            </button>
            {filterPersonaList.length > 1 && (
              <>
                <div className="my-1 border-t border-line" />
                <div className="px-2 pt-1 pb-1 flex items-center justify-between">
                  <span className="text-[10.5px] uppercase tracking-[0.06em] text-faint font-semibold">
                    Filter by coworker
                  </span>
                  {filterPersonas.size > 0 && (
                    <button className="text-[11px] text-accent" onClick={() => setFilterPersonas(new Set())}>
                      Clear
                    </button>
                  )}
                </div>
                <div className="max-h-52 overflow-y-auto">
                  {filterPersonaList.map((p) => {
                    const checked = filterPersonas.has(p.id);
                    return (
                      <button
                        key={p.id}
                        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-[13px] text-left hover:bg-paper"
                        onClick={() => toggleFilterPersona(p.id)}
                      >
                        <span
                          className={
                            "w-3.5 h-3.5 rounded border grid place-items-center shrink-0 text-white " +
                            (checked ? "bg-accent border-accent" : "border-line")
                          }
                        >
                          {checked && <span className="text-[9px] leading-none">✓</span>}
                        </span>
                        <span className="flex-1 truncate">{p.name}</span>
                      </button>
                    );
                  })}
                </div>
                <div className="px-2 pt-1 pb-0.5 text-[11px] text-faint leading-snug">
                  None checked shows all.
                </div>
              </>
            )}
          </div>
        </>
      )}
      {magicSortPanel()}
    </div>
    );
  };

  // Code/Cowork group by project; Chat is a flat recents list.
  const byProject = useMemo(() => {
    const grouped = new Map<string, SessionInfo[]>();
    for (const s of mine) {
      if (!grouped.has(s.workspace)) grouped.set(s.workspace, []);
      grouped.get(s.workspace)!.push(s);
    }
    return grouped;
  }, [mine]);

  const filteredByProject = useMemo(() => {
    const grouped = new Map<string, SessionInfo[]>();
    for (const [proj, list] of byProject) grouped.set(proj, list.filter(matches));
    return grouped;
  }, [byProject, normalizedQuery]);

  // Projects are tracked PER SURFACE: a folder appears under Code only if it has Code sessions,
  // under Cowork only if it has Cowork sessions (+ the currently-open folder). No cross-bleed.
  const projectOrder: string[] = [];
  const seen = new Set<string>();
  // Pin the active folder at top only when browsing the active persona (else it belongs elsewhere).
  if (props.workspace && browseKey === props.agent) {
    projectOrder.push(props.workspace);
    seen.add(props.workspace);
  }
  for (const s of mine) {
    if (s.workspace && !seen.has(s.workspace)) {
      seen.add(s.workspace);
      projectOrder.push(s.workspace);
    }
  }

  // Surfaced + enabled personas drive the surface list (default persona first); fall back to the
  // static set until loaded. A persona that has live sessions ALWAYS gets a section, surfaced or
  // not — every session must have a home in the grouped layout (a picker preference can hide the
  // persona from New Session, never orphan its conversations).
  const agentsWithSessions = new Set(
    props.sessions
      .filter((s) => !s.archived && !s.session_id.startsWith("__"))
      .map((s) => s.agent),
  );
  const visibleSurfaces = (
    personas
      ? personas
          .filter((p) => (p.enabled && p.surfaced) || agentsWithSessions.has(p.id))
          .sort((a, b) => Number(b.default) - Number(a.default)) // default leads
          .map(surfaceFromPersona)
      : SURFACES.filter(
          (s) => s.key === "cowork" || props.surfaces[s.key as keyof SurfaceVisibility],
        )
  ).filter((s) => personaVisible(s.key));

  const isCurrent = (key: string) => props.agent === key; // the active session's persona
  const isExpanded = (key: string) => openKey === key; // its body is open
  // Expand ≠ switch: clicking a header only browses (toggles the accordion). The chat area
  // changes only when a session is selected or "New session" is clicked.
  const onHeaderClick = (key: string) => setOpenKey((k) => (k === key ? null : key));

  // The expanded body for the active surface: a "New session" action, then the project-grouped
  // (or flat) session list, then the archived disclosure.
  const surfaceBody = () => {
    return (
      <div className="space-y-1 px-1.5 pb-2 pt-0.5">
        {/* Body is flush inside the expanded group's fill (provided by the wrapper) so the header +
            its sessions read as one connected block — clear where a group ends and the next begins. */}
        {/* No per-persona "New session" here — the top split button's ▾ already starts a session
            in any persona (it was redundant + the mock's grouped cards don't have it). */}
        {workspaceSurface ? (
          <>
            {/* Codex-style Projects: a "+" header affordance, then collapsible folders whose
                rows carry a right-aligned compact age and truncate to PROJECT_PEEK + "Show more". */}
            <div className="flex items-center justify-between px-1.5 pt-1">
              <span className="text-[10.5px] uppercase tracking-[0.07em] text-faint font-semibold">
                Projects
              </span>
              <button
                className="w-5 h-5 grid place-items-center rounded text-faint hover:text-ink hover:bg-panel"
                title="New project"
                aria-label="New project"
                onClick={() => props.onNewProject(browseKey)}
              >
                <Icon name="folderPlus" size={14} />
              </button>
            </div>
            <div className="space-y-0.5">
              {projectOrder.length === 0 && (
                <div className="px-2 py-1.5 text-[12px] text-faint leading-snug">
                  No projects yet — start one with the + above.
                </div>
              )}
              {projectOrder.map((proj) => {
                const list = filteredByProject.get(proj) || [];
                if (normalizedQuery && list.length === 0) return null; // hide non-matching folders while searching
                const isActive = proj === props.workspace;
                // Open the active project by default; if none is active (browsing from another
                // persona), open the most-recent folder so the accordion isn't all-collapsed.
                const activeInOrder = !!props.workspace && projectOrder.includes(props.workspace);
                const defaultOpen = isActive || (!activeInOrder && proj === projectOrder[0]);
                const open = !!normalizedQuery || defaultOpen !== projToggled.has(proj);
                const showAll = !!normalizedQuery || projShowAll.has(proj);
                const shown = showAll ? list : list.slice(0, peek);
                return (
                  <div key={proj}>
                    <div
                      className={
                        "flex items-center gap-1.5 px-1.5 py-1 rounded-lg cursor-pointer select-none hover:bg-panel " +
                        (isActive ? "text-ink" : "text-muted hover:text-ink")
                      }
                      onClick={() => setProjToggled((s) => toggleSet(s, proj))}
                      title={proj}
                    >
                      <Icon name="folder" size={15} className="shrink-0" />
                      <span
                        className={
                          "truncate min-w-0 text-[12.5px] " + (isActive ? "font-semibold" : "font-medium")
                        }
                      >
                        {baseName(proj)}
                      </span>
                      {/* Disclosure chevron sits AFTER the name (Codex parity), not leading the row. */}
                      <Icon
                        name={open ? "chevronDown" : "chevronRight"}
                        size={12}
                        className="text-faint shrink-0"
                      />
                    </div>
                    {open &&
                      (list.length > 0 ? (
                        // pl-[19px] aligns each session's name under the folder NAME (folder icon
                        // 15 + gap 6 + row px 6 − session px 8 = 19), per Rohit's clean-column ask.
                        <div className="space-y-0.5 pl-[19px]">
                          {shown.map((s) => sessionRow(s, { showTime: true }))}
                          {!showAll && list.length > peek && (
                            <button
                              className="px-2 py-1 text-[12px] text-faint hover:text-muted"
                              onClick={() => setProjShowAll((s) => toggleSet(s, proj))}
                            >
                              Show more ({list.length - peek})
                            </button>
                          )}
                        </div>
                      ) : (
                        <div className="px-2 py-1.5 pl-[19px] text-[12px] text-faint leading-snug">
                          No conversations in this project yet.
                        </div>
                      ))}
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <div className="space-y-0.5">
            {mine.filter(matches).length === 0 ? (
              <div className="px-2 py-1.5 text-[12px] text-faint leading-snug">
                {normalizedQuery ? "No matching conversations." : "No conversations yet."}
              </div>
            ) : (
              <>
                {(personaShowAll.has(browseKey)
                  ? mine.filter(matches)
                  : mine.filter(matches).slice(0, peek)
                ).map((s) => sessionRow(s))}
                {!personaShowAll.has(browseKey) && mine.filter(matches).length > peek && (
                  <button
                    className="px-2 py-1 text-[12px] text-faint hover:text-muted"
                    onClick={() => setPersonaShowAll((s) => toggleSet(s, browseKey))}
                  >
                    Show more ({mine.filter(matches).length - peek})
                  </button>
                )}
              </>
            )}
          </div>
        )}

      </div>
    );
  };

  return (
    <div
      className="sidebar flex flex-col min-h-0 bg-panel border-r border-line"
      onMouseLeave={props.onPeekLeave}
    >
      {/* Header: collapse/pin control FIRST + wordmark. The pin sits at the same screen position
          as the collapsed reveal button (see .nav-pin-btn / .nav-reveal-btn in styles.css), so
          hovering the reveal peeks the nav and the pin lands right under the cursor — no travel.
          data-tauri-drag-region drags the window; on desktop the row clears the traffic lights. */}
      <div className="brand px-3.5 pt-2.5 pb-2 flex items-center gap-2" data-tauri-drag-region>
        {/* Collapse (dock) / pin the sidebar. ⌘B mirrors this. */}
        {props.onCollapse && (
          <button
            className="nav-pin-btn w-7 h-7 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper shrink-0"
            title={props.collapsed ? "Dock sidebar (⌘B)" : "Collapse sidebar (⌘B)"}
            aria-label={props.collapsed ? "Dock sidebar" : "Collapse sidebar"}
            onClick={props.onCollapse}
          >
            <Icon name="sidebar" size={16} />
          </button>
        )}
        <div className="brand-wordmark text-[15px]">OpenWorker<span className="beta-tag">BETA</span></div>
      </div>

      {/* New session: split button — primary starts the last-used persona; ▾ picks a specific one. */}
      <NewSessionSplit
        personas={personas}
        current={props.agent}
        onNew={props.onNewSession}
        onNewFolder={() => {
          setGroupBy("folder");
          startCreatingFolder(null);
        }}
        onManage={props.onManagePersonas}
      />

      {/* Search: a borderless nav-style entry (not a boxed input) that opens the command-palette
          SearchModal over the whole app. Matches the bottom-nav rows to reduce the boxy look. */}
      <div className="px-2.5 mt-1">
        <button
          className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] text-left text-muted hover:bg-paper hover:text-ink"
          onClick={() => setSearchModalOpen(true)}
        >
          <Icon name="search" size={15} className="shrink-0" /> Search
        </button>
      </div>

      <div className="px-2.5 mt-1">
        <button
          className={
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] text-left hover:bg-paper hover:text-ink " +
            (props.projectsActive ? "text-ink bg-paper" : "text-muted")
          }
          data-testid="nav-projects"
          onClick={() => props.onOpenProjects()}
        >
          <Icon name="folder" size={15} className="shrink-0" />
          <span className="flex-1">Projects</span>
          <ProjectCountBadge n={topicProjects.length} />
        </button>
      </div>

      {/* Automations: a first-class nav row (UX-023) — the account menu keeps its entry.
          The badge is the cross-automation unseen-run total. */}
      <div className="px-2.5 mt-1">
        <button
          className={
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] text-left hover:bg-paper hover:text-ink " +
            (props.scheduledActive ? "text-ink bg-paper" : "text-muted")
          }
          data-testid="nav-automations"
          onClick={props.onOpenScheduled}
        >
          <Icon name="clock" size={15} className="shrink-0" />
          <span className="flex-1">Automations</span>
        </button>
      </div>

      {/* Scroll area: Pinned band + the RECENT header (with group/filter control), then the body —
          grouped (per-persona accordion) or flat (chronological list). */}
      <div className="flex-1 overflow-y-auto px-2.5 mt-3 pb-2">
        <div className="space-y-4">
          {pinnedBand()}
          {scheduledBand()}
          {projectsBand()}
          <div>
            {recentHeader()}
            {layout === "grouped" ? (
            <div className="space-y-1.5">
              {visibleSurfaces.map((s) => {
                const expanded = isExpanded(s.key);
                return (
                  // When expanded, the wrapper carries the recessed fill so the header sits INSIDE
                  // the block with its sessions (one connected group). Collapsed = a plain row.
                  <div
                    key={s.key}
                    className={expanded ? "rounded-xl bg-paper/70 overflow-hidden" : ""}
                  >
                    <div
                      className={
                        "flex items-center gap-2.5 px-2 py-2 cursor-pointer select-none " +
                        (expanded
                          ? ""
                          : isCurrent(s.key)
                            ? "rounded-lg bg-paper"
                            : "rounded-lg hover:bg-paper")
                      }
                      onClick={() => onHeaderClick(s.key)}
                    >
                      <span
                        className={
                          "min-w-0 flex-1 truncate text-[13px] " +
                          (isCurrent(s.key) ? "font-semibold text-ink" : "font-medium text-ink")
                        }
                      >
                        {s.label}
                      </span>
                      <LiveDot state={liveByPersona.get(s.key)} />
                      <AttnBadge n={attnByPersona.get(s.key) || 0} />
                      {/* Persona configuration moved to Settings ▸ Personas (Rohit's call
                          2026-07-07) — the per-group gear read as clutter here. */}
                      <Icon
                        name={expanded ? "chevronDown" : "chevronRight"}
                        size={15}
                        className="text-faint shrink-0"
                      />
                    </div>
                    {expanded && surfaceBody()}
                  </div>
                );
              })}
            </div>
            ) : layout === "folder" ? (
            <div className="space-y-1.5" data-testid="folder-layout">
              {creatingFolderForSession === null ? (
                <div className="w-full flex items-center px-2 py-1 rounded-lg">
                  <input
                    className="flex-1 min-w-0 px-1.5 py-0.5 rounded-md bg-panel border border-accent text-[13px] text-ink outline-none"
                    value={folderEditValue}
                    placeholder="Folder name"
                    autoFocus
                    onFocus={(event) => event.currentTarget.select()}
                    onChange={(event) => setFolderEditValue(event.target.value)}
                    onBlur={() => void createChatFolder(folderEditValue)}
                    onKeyDown={(event) => {
                      event.stopPropagation();
                      if (event.key === "Enter") void createChatFolder(folderEditValue);
                      else if (event.key === "Escape") cancelFolderEdit();
                    }}
                  />
                </div>
              ) : (
                <button
                  className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-[12.5px] text-muted hover:text-ink hover:bg-paper active:bg-line/60"
                  onClick={() => startCreatingFolder(null)}
                >
                  <Icon name="folderPlus" size={14} className="shrink-0" />
                  <span>New folder</span>
                </button>
              )}
              {folders.map((folder) => {
                const sessions = sessionsForFolder(folder.id);
                const expanded = !collapsedFolders.has(folder.id);
                const editing = editingFolderId === folder.id;
                return (
                  <div
                    key={folder.id}
                    className={expanded ? "rounded-xl bg-paper/70 overflow-hidden" : ""}
                  >
                    <div
                      className={
                        "flex items-center gap-2 px-2 py-2 cursor-pointer select-none " +
                        (expanded ? "" : "rounded-lg hover:bg-paper active:bg-line/60")
                      }
                      onClick={() => {
                        if (!editing) {
                          setCollapsedFolders((current) => toggleSet(current, folder.id));
                        }
                      }}
                    >
                      {editing ? (
                        <input
                          className="flex-1 min-w-0 px-1.5 py-0.5 rounded-md bg-panel border border-accent text-[13px] text-ink outline-none"
                          value={folderEditValue}
                          autoFocus
                          onFocus={(event) => event.currentTarget.select()}
                          onClick={(event) => event.stopPropagation()}
                          onChange={(event) => setFolderEditValue(event.target.value)}
                          onBlur={() => void renameChatFolder(folder)}
                          onKeyDown={(event) => {
                            event.stopPropagation();
                            if (event.key === "Enter") void renameChatFolder(folder);
                            else if (event.key === "Escape") cancelFolderEdit();
                          }}
                        />
                      ) : (
                        <>
                          <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">
                            {folder.name}
                          </span>
                          <LiveDot state={folderLiveness(sessions)} />
                          <AttnBadge n={folderAttention(sessions)} />
                          {folderActions(folder)}
                          <Icon
                            name={expanded ? "chevronDown" : "chevronRight"}
                            size={15}
                            className="text-faint shrink-0"
                          />
                        </>
                      )}
                    </div>
                    {expanded && (
                      <div className="px-1.5 pb-1.5 space-y-0.5">
                        {sessions.length > 0 ? (
                          sessions.map((session) => sessionRow(session, { showTime: true }))
                        ) : (
                          <div className="px-2 py-1.5 text-[12px] text-faint">No conversations</div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
              {(() => {
                const expanded = !collapsedFolders.has(UNFILED_KEY);
                return (
                  <div
                    className={expanded ? "rounded-xl bg-paper/70 overflow-hidden" : ""}
                    data-testid="folder-unfiled"
                  >
                    <div
                      className={
                        "flex items-center gap-2 px-2 py-2 cursor-pointer select-none " +
                        (expanded ? "" : "rounded-lg hover:bg-paper active:bg-line/60")
                      }
                      onClick={() => setCollapsedFolders((current) => toggleSet(current, UNFILED_KEY))}
                    >
                      <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">
                        Unfiled
                      </span>
                      <LiveDot state={folderLiveness(unfiledSessions)} />
                      <AttnBadge n={folderAttention(unfiledSessions)} />
                      <Icon
                        name={expanded ? "chevronDown" : "chevronRight"}
                        size={15}
                        className="text-faint shrink-0"
                      />
                    </div>
                    {expanded && (
                      <div className="px-1.5 pb-1.5 space-y-0.5">
                        {unfiledSessions.length > 0 ? (
                          unfiledSessions.map((session) => sessionRow(session, { showTime: true }))
                        ) : (
                          <div className="px-2 py-1.5 text-[12px] text-faint">No conversations</div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })()}
              {archiveFolder()}
            </div>
            ) : (
            <div className="space-y-1.5">
              <div className="space-y-0.5">
              {recentSessions.length === 0 ? (
                <div className="px-2 py-1.5 text-[12px] text-faint leading-snug">
                  {normalizedQuery ? "No matching conversations." : "No conversations yet."}
                </div>
              ) : (
                <>
                  {(recentExpanded
                    ? recentSessions
                    : recentSessions.slice(0, RECENT_PEEK)
                  ).map((s) => cardRow(s))}
                  {recentSessions.length > RECENT_PEEK && (
                    <button
                      className="w-full text-left px-2 py-1.5 text-[12px] text-muted hover:text-ink"
                      onClick={() => setRecentExpanded((v) => !v)}
                    >
                      {recentExpanded
                        ? "Show less"
                        : `Show ${recentSessions.length - RECENT_PEEK} more`}
                    </button>
                  )}
                </>
              )}
              </div>
              {archiveFolder()}
            </div>
            )}
            {layout === "grouped" && <div className="mt-1.5">{archiveFolder()}</div>}
          </div>
        </div>
      </div>

      {/* Bottom (§26): exactly ONE row — the account anchor. The inbox chip on it is
          state-driven with a sticky unlock (quiet when empty, accent + count when pending);
          everything else lives in the account menu, which ALWAYS lists Inbox + Connectors. */}
      <div className="px-2.5 py-2 border-t border-line">
        <div className="relative">
          {appMenuOpen && (
            <>
              <div className="fixed inset-0 z-30" onClick={() => setAppMenuOpen(false)} />
              <div
                className="absolute z-40 bottom-full left-0 right-0 mb-1 rounded-xl border border-line bg-panel shadow-2xl py-1"
                data-testid="account-menu"
                role="menu"
              >
                {cloud?.signed_in ? (
                  <div
                    className="px-3 py-1.5 mb-1 text-[11px] text-faint truncate border-b border-line"
                    title={`${accountEmail} · OpenWorker Cloud`}
                  >
                    {accountEmail} · OpenWorker Cloud
                  </div>
                ) : (
                  <>
                    <div className="px-3 py-1.5 text-[11px] text-faint border-b border-line">
                      Not signed in — one-click connections need OpenWorker Cloud
                    </div>
                    <button
                      className="w-full flex items-center gap-2.5 px-3 py-1.5 mb-1 text-[13px] text-left text-accent hover:bg-paper"
                      data-testid="account-sign-in"
                      onClick={async () => {
                        setAppMenuOpen(false);
                        // Opens the system browser server-side; completion lands out-of-band,
                        // so poll until it flips (refocusing the window also refetches).
                        await cloudLogin().catch(() => {});
                        waitForCloudSignIn((s) => {
                          if (s) setCloud(s);
                          // Other always-mounted consumers (Settings' telemetry card,
                          // connector panes) refetch on this.
                          if (s?.signed_in) announceCloudChanged();
                        });
                      }}
                    >
                      <Icon name="plug" size={15} className="shrink-0" /> Sign in to OpenWorker
                      Cloud
                    </button>
                  </>
                )}
                {appMenuItem(
                  "inbox",
                  "Inbox",
                  props.onOpenInbox,
                  props.inboxActive,
                  <AttnBadge n={totalAttention} />,
                )}
                {appMenuItem("plug", "Connectors", props.onOpenIntegrations, props.integrationsActive)}
                <div className="h-px bg-line my-1 mx-2" />
                {appMenuItem(
                  "gear",
                  "Settings",
                  props.onManage,
                  false,
                  <span className="text-[11px] text-faint">⌘ ,</span>,
                )}
                {appMenuItem("clock", "Automations", props.onOpenScheduled, props.scheduledActive)}
                {appMenuItem("audit", "Activity", props.onOpenAudit, props.auditActive)}
                {cloud?.signed_in && (
                  <>
                    <div className="h-px bg-line my-1 mx-2" />
                    {appMenuItem("signOut", "Sign out", async () => {
                      await cloudLogout().catch(() => {});
                      announceCloudChanged();
                    })}
                  </>
                )}
              </div>
            </>
          )}

          <button
            className={
              "w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-[13px] text-left " +
              (appMenuOpen ? "bg-paper text-ink" : "hover:bg-paper")
            }
            data-testid="account-row"
            onClick={() => {
              if (!appMenuOpen) refreshCloud();
              setAppMenuOpen((v) => !v);
            }}
            aria-haspopup="menu"
            aria-expanded={appMenuOpen}
            aria-label={cloud?.signed_in ? `Account: ${accountEmail}` : "Account: not signed in"}
          >
            <span
              className={
                "w-6 h-6 rounded-full grid place-items-center text-[10.5px] font-semibold shrink-0 " +
                (cloud?.signed_in
                  ? "bg-accentSoft text-accent"
                  : "bg-paper text-faint border border-line")
              }
              aria-hidden
            >
              {cloud?.signed_in ? accountName.slice(0, 1).toUpperCase() : "?"}
            </span>
            <span className={"truncate " + (cloud?.signed_in ? "" : "text-muted")}>
              {cloud?.signed_in ? accountName : "Not signed in"}
            </span>
            {cloud?.signed_in && (
              <span
                className="w-[7px] h-[7px] rounded-full bg-ok shrink-0"
                title="Signed in to OpenWorker Cloud"
                aria-hidden
              />
            )}
            <span className="flex-1" />
            {inboxUnlocked && (
              <span
                className={
                  "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11.5px] shrink-0 cursor-pointer " +
                  (totalAttention > 0
                    ? "bg-accentSoft text-accent font-semibold"
                    : "text-faint hover:text-ink")
                }
                data-testid="inbox-chip"
                role="button"
                aria-label={
                  totalAttention > 0 ? `Inbox — ${totalAttention} items need you` : "Inbox"
                }
                title={totalAttention > 0 ? `Inbox — ${totalAttention} items need you` : "Inbox"}
                onClick={(e) => {
                  // The chip goes STRAIGHT to Inbox — the menu is the row's target, not the chip's.
                  e.stopPropagation();
                  setAppMenuOpen(false);
                  props.onOpenInbox();
                }}
              >
                <Icon name="inbox" size={13} />
                {totalAttention > 0 ? totalAttention : null}
              </span>
            )}
            <Icon
              name="chevronDown"
              size={14}
              className={"text-faint shrink-0 transition-transform " + (appMenuOpen ? "" : "rotate-180")}
            />
          </button>
        </div>
      </div>

      {searchModalOpen && (
        <SearchModal
          sessions={props.sessions}
          personas={personas ?? undefined}
          onSelect={(id, ws, ag) => {
            setSearchModalOpen(false);
            props.onSelectSession(id, ws, ag);
          }}
          onClose={() => setSearchModalOpen(false)}
        />
      )}
    </div>
  );
}

// New-session split button (§8): the primary action starts a session with the last-used persona
// (`current`); the ▾ opens a menu of the enabled personas (from /v1/personas) plus a "Manage
// personas…" entry. A plain custom split control — the pill-shaped Dropdown doesn't fit this shape.
function NewSessionSplit({
  personas,
  current,
  onNew,
  onNewFolder,
  onManage,
}: {
  personas: Persona[] | null;
  current: string;
  onNew: (agent: string) => void;
  onNewFolder: () => void;
  onManage: () => void;
}) {
  const [open, setOpen] = useState(false);
  const enabled = (personas || []).filter((p) => p.enabled);
  // A single enabled persona does not need a redundant picker, but the menu still carries the
  // folder action. `personas === null` (still loading) keeps the persona section until resolved.
  const solo = personas !== null && enabled.length <= 1;
  return (
    <div className="px-3 pt-2 relative">
      <div className="flex">
        <button
          className="newsplit-primary min-h-10 flex-1 text-left px-3 py-2 rounded-l-lg bg-accent text-white text-[13px] font-medium hover:opacity-90 active:opacity-80 flex items-center gap-2"
          onClick={() => onNew(solo && enabled.length === 1 ? enabled[0].id : current)}
        >
          <Icon name="plus" size={15} className="shrink-0" /> New session
        </button>
        <button
          className="w-10 min-h-10 rounded-r-lg bg-accent text-white border-l border-white/25 hover:opacity-90 active:opacity-80 grid place-items-center"
          title="New session options"
          aria-label="New session options"
          onClick={() => setOpen((v) => !v)}
        >
          <Icon name="chevronDown" size={13} />
        </button>
      </div>
      {open && (
        <>
          <div className="fixed inset-0 z-20" onClick={() => setOpen(false)} />
          <div className="newsplit-menu absolute left-3 right-3 mt-1 z-30 bg-panel border border-line rounded-xl2 shadow-xl p-1">
            {!solo && (
              <>
                <div className="px-2 py-1 text-[10.5px] uppercase tracking-[0.06em] text-faint font-semibold">
                  Start a session as
                </div>
                {enabled.map((p) => (
                  <button
                    key={p.id}
                    className="w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg hover:bg-paper active:bg-line/60 text-left"
                    onClick={() => {
                      setOpen(false);
                      onNew(p.id);
                    }}
                  >
                    <span className="w-6 h-6 rounded-md bg-paper border border-line grid place-items-center text-muted shrink-0">
                      <PersonaGlyph icon={p.icon} family={p.family} size={12} />
                    </span>
                    <span className="min-w-0">
                      <span className="block text-[13px] font-medium truncate">
                        {shortPersonaName(p.name, p.id)}
                      </span>
                      {p.tagline && (
                        <span className="block text-[11px] text-muted truncate">{p.tagline}</span>
                      )}
                    </span>
                  </button>
                ))}
                <div className="h-px bg-line my-1 mx-2" />
              </>
            )}
            <button
              className="w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg text-[12.5px] text-muted hover:text-ink hover:bg-paper active:bg-line/60 text-left"
              onClick={() => {
                setOpen(false);
                onNewFolder();
              }}
            >
              <span className="w-6 h-6 grid place-items-center shrink-0">
                <Icon name="folderPlus" size={14} />
              </span>
              <span>New folder</span>
            </button>
            {showPersonas() && (
              <div className="border-t border-line mt-1 pt-1">
                <button
                  className="w-full px-2 py-1.5 rounded-lg hover:bg-paper text-left text-[12.5px] text-muted"
                  onClick={() => {
                    setOpen(false);
                    onManage();
                  }}
                >
                  Manage personas…
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
