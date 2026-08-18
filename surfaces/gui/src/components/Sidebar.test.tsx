import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { Sidebar } from "./Sidebar";
import type { SessionInfo } from "../types";

// Hermetic fetch stub routing by URL substring + method; records calls for POST assertions.
type Call = { url: string; method: string; body: any };

function stubFetch(routes: { match: string; method?: string; json: any }[]) {
  const calls: Call[] = [];
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    const method = (init?.method || "GET").toUpperCase();
    calls.push({ url, method, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    for (const r of routes) {
      if (url.includes(r.match) && (!r.method || r.method === method)) {
        return { ok: true, json: async () => r.json } as Response;
      }
    }
    return { ok: true, json: async () => ({}) } as Response;
  });
  vi.stubGlobal("fetch", fn);
  return calls;
}

const PERSONAS = {
  personas: [
    { id: "cowork", name: "OpenWorker", icon: "cowork", tagline: "general assistant", family: "knowledge", enabled: true, surfaced: true, default: true },
    { id: "ops", name: "Ops", icon: "ops", tagline: "incidents, runbooks", family: "code", enabled: true, surfaced: true, default: false },
    { id: "code", name: "Code", icon: "code", tagline: "repository work", family: "code", enabled: true, surfaced: true, default: false },
    { id: "secret", name: "Disabled One", icon: "cowork", tagline: "off", family: "knowledge", enabled: false, surfaced: false, default: false },
  ],
};

const SESSIONS: SessionInfo[] = [
  { session_id: "s-ops-1", title: "incident watch", workspace: "/w", agent: "ops", model: "m", mode: "interactive", updated_at: "2026-06-29", messages: 2 },
  { session_id: "s-cowork-1", title: "hi there", workspace: "", agent: "cowork", model: "m", mode: "interactive", updated_at: "2026-06-29", messages: 1 },
];

const baseProps = {
  agent: "cowork",
  workspace: "",
  surfaces: { cowork: true, chat: false, code: false },
  sessions: SESSIONS,
  projects: [],
  activeSession: "s-cowork-1",
  onSwitchAgent: vi.fn(),
  onNewSession: vi.fn(),
  onSelectSession: vi.fn(),
  onNewProject: vi.fn(),
  onRenameSession: vi.fn(),
  onDeleteSession: vi.fn(),
  onArchiveSession: vi.fn(),
  onTogglePin: vi.fn(),
  onSetSessionFolder: vi.fn(),
  onApplyMagicSort: vi.fn(async () => ({ ok: true, moved: 0 })),
  onArchiveAllSessions: vi.fn(),
  onDeleteFolder: vi.fn(async () => true),
  onManage: vi.fn(),
  onOpenPersona: vi.fn(),
  onManagePersonas: vi.fn(),
  onOpenProjects: vi.fn(),
  onOpenScheduled: vi.fn(),
  onOpenAutomation: vi.fn(),
  onOpenIntegrations: vi.fn(),
  onOpenAudit: vi.fn(),
  onOpenInbox: vi.fn(),
  scheduledActive: false,
  projectsActive: false,
  integrationsActive: false,
  auditActive: false,
  inboxActive: false,
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("Sidebar group/filter control", () => {
  it("choosing Persona persists via setNavLayout and switches to the per-persona accordion", async () => {
    const calls = stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
      { match: "/v1/settings/nav-layout", method: "POST", json: { ok: true, nav_layout: "grouped" } },
    ]);
    render(<Sidebar {...baseProps} />);

    // personas load drives the surfaces; the RECENT header's group/filter control is always present.
    const control = await screen.findByLabelText("Group and filter conversations");

    // Open the popover and choose "Group by → Persona".
    fireEvent.click(control);
    fireEvent.click(await screen.findByText("Persona"));

    // POSTs the new layout pref.
    await waitFor(() => {
      const post = calls.find((c) => c.method === "POST" && c.url.includes("/v1/settings/nav-layout"));
      expect(post).toBeTruthy();
      expect(post!.body).toMatchObject({ nav_layout: "grouped" });
    });

    // Close the popover (it stays open so you can group AND filter in one visit) before asserting
    // the accordion — otherwise "Ops" also matches the filter-by-coworker checkbox.
    fireEvent.click(control);

    // Grouped view = the per-persona accordion. The Ops header appears; expanding it lists its
    // session. (Persona configuration moved to Settings ▸ Personas, so there is no header gear.)
    const opsHeader = await screen.findByText("Ops");
    fireEvent.click(opsHeader);
    expect(screen.getByText("incident watch")).toBeTruthy();
    expect(screen.queryByTitle("About the Ops persona")).toBeNull();
  });

  it("offers Folder grouping and renders named folders followed by Unfiled", async () => {
    const calls = stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
      {
        match: "/v1/folders",
        method: "GET",
        json: { folders: [{ id: "work", name: "Work", created: "2026-08-17T00:00:00Z" }] },
      },
      { match: "/v1/settings/nav-layout", method: "POST", json: { ok: true, nav_layout: "folder" } },
    ]);
    const sessions = [
      { ...SESSIONS[0], folder_id: "work" },
      SESSIONS[1],
    ];
    render(<Sidebar {...baseProps} sessions={sessions} />);

    fireEvent.click(await screen.findByLabelText("Group and filter conversations"));
    fireEvent.click(await screen.findByText("Folder"));

    await waitFor(() =>
      expect(calls.some((call) => call.body?.nav_layout === "folder")).toBe(true),
    );
    expect(await screen.findByTestId("folder-layout")).toBeTruthy();
    expect(screen.getByText("Work")).toBeTruthy();
    expect(screen.getByText("incident watch")).toBeTruthy();
    const unfiled = screen.getByTestId("folder-unfiled");
    expect(within(unfiled).getByText("hi there")).toBeTruthy();
    expect(unfiled.parentElement?.lastElementChild).toBe(unfiled);
  });

  it("keeps sessions with a deleted folder reference under Unfiled", async () => {
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "folder" } },
      {
        match: "/v1/folders",
        method: "GET",
        json: { folders: [{ id: "work", name: "Work", created: "2026-08-17T00:00:00Z" }] },
      },
    ]);
    render(<Sidebar {...baseProps} sessions={[{ ...SESSIONS[0], folder_id: "deleted-folder" }]} />);

    const unfiled = await screen.findByTestId("folder-unfiled");
    expect(within(unfiled).getByText("incident watch")).toBeTruthy();
  });
});

describe("Magic sort preview", () => {
  it("previews proposed moves, lets one row opt out, then applies the rest", async () => {
    const onApplyMagicSort = vi.fn(async (_proposals: unknown[]) => ({
      ok: true,
      moved: 1,
      folders_created: 1,
    }));
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
      { match: "/v1/folders", method: "GET", json: { folders: [] } },
      {
        match: "/v1/magic-sort/propose",
        method: "POST",
        json: {
          ok: true,
          considered: 2,
          skipped: 0,
          proposals: [
            { session_id: "s-ops-1", title: "incident watch", action: "new_folder", target_name: "Operations" },
            { session_id: "s-cowork-1", title: "hi there", action: "leave", target_name: "Leave where it is" },
          ],
        },
      },
    ]);
    render(<Sidebar {...baseProps} onApplyMagicSort={onApplyMagicSort} />);

    fireEvent.click(await screen.findByLabelText("Group and filter conversations"));
    fireEvent.click(screen.getByTestId("magic-sort-menu-item"));

    const panel = await screen.findByTestId("magic-sort-panel");
    expect(within(panel).getByText("Review Magic sort")).toBeTruthy();
    expect(within(panel).getByText("Operations")).toBeTruthy();
    expect(within(panel).getByText("new")).toBeTruthy();
    fireEvent.click(within(panel).getByLabelText("Exclude hi there"));
    fireEvent.click(within(panel).getByText("Apply 1"));

    await waitFor(() => expect(onApplyMagicSort).toHaveBeenCalledTimes(1));
    expect(onApplyMagicSort.mock.calls[0][0]).toEqual([
      { session_id: "s-ops-1", title: "incident watch", action: "new_folder", target_name: "Operations" },
    ]);
    await waitFor(() => expect(screen.queryByTestId("magic-sort-panel")).toBeNull());
  });

  it("shows a calm empty state when there are no useful moves", async () => {
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
      { match: "/v1/folders", method: "GET", json: { folders: [] } },
      {
        match: "/v1/magic-sort/propose",
        method: "POST",
        json: { ok: true, considered: 2, skipped: 0, proposals: [] },
      },
    ]);
    render(<Sidebar {...baseProps} />);

    fireEvent.click(await screen.findByLabelText("Group and filter conversations"));
    fireEvent.click(screen.getByTestId("magic-sort-menu-item"));

    expect(await screen.findByText("Nothing to sort")).toBeTruthy();
    expect(screen.getByText("Your recent chats already look tidy.")).toBeTruthy();
  });
});

describe("Chat folder inline naming", () => {
  const renderFolderLayout = async (extraRoutes: Parameters<typeof stubFetch>[0] = []) => {
    const calls = stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "folder" } },
      {
        match: "/v1/folders",
        method: "GET",
        json: { folders: [{ id: "work", name: "Work", created: "2026-08-17T00:00:00Z" }] },
      },
      ...extraRoutes,
    ]);
    render(<Sidebar {...baseProps} />);
    await screen.findByTestId("folder-layout");
    return calls;
  };

  it("creates a folder from the inline input on Enter", async () => {
    const calls = await renderFolderLayout([
      {
        match: "/v1/folders",
        method: "POST",
        json: { ok: true, folder: { id: "plans", name: "Plans", created: "2026-08-17T00:00:00Z" } },
      },
    ]);

    fireEvent.click(screen.getByText("New folder"));
    const input = screen.getByPlaceholderText("Folder name");
    fireEvent.change(input, { target: { value: "Plans" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() =>
      expect(calls.some((call) => call.method === "POST" && call.body?.name === "Plans")).toBe(true),
    );
    expect(await screen.findByText("Plans")).toBeTruthy();
    expect(screen.queryByPlaceholderText("Folder name")).toBeNull();
  });

  it("cancels folder creation on Escape", async () => {
    const calls = await renderFolderLayout();

    fireEvent.click(screen.getByText("New folder"));
    const input = screen.getByPlaceholderText("Folder name");
    fireEvent.change(input, { target: { value: "Discard me" } });
    fireEvent.keyDown(input, { key: "Escape" });

    expect(screen.queryByPlaceholderText("Folder name")).toBeNull();
    expect(calls.filter((call) => call.method === "POST" && call.url.includes("/v1/folders"))).toHaveLength(0);
  });

  it("renames a folder from its pre-filled inline input on Enter", async () => {
    const calls = await renderFolderLayout([
      {
        match: "/v1/folders/work/rename",
        method: "POST",
        json: { ok: true, folder: { id: "work", name: "Focus", created: "2026-08-17T00:00:00Z" } },
      },
    ]);

    fireEvent.click(screen.getByLabelText("Folder actions for Work"));
    fireEvent.click(screen.getByText("Rename"));
    const input = screen.getByDisplayValue("Work");
    fireEvent.change(input, { target: { value: "Focus" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() =>
      expect(calls.some((call) => call.url.includes("/v1/folders/work/rename") && call.body?.name === "Focus")).toBe(true),
    );
    expect(await screen.findByText("Focus")).toBeTruthy();
  });

  it("does not create or rename a folder with a whitespace-only name", async () => {
    const calls = await renderFolderLayout();

    fireEvent.click(screen.getByText("New folder"));
    let input = screen.getByPlaceholderText("Folder name");
    fireEvent.change(input, { target: { value: "   " } });
    fireEvent.keyDown(input, { key: "Enter" });

    fireEvent.click(screen.getByLabelText("Folder actions for Work"));
    fireEvent.click(screen.getByText("Rename"));
    input = screen.getByDisplayValue("Work");
    fireEvent.change(input, { target: { value: "   " } });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(calls.filter((call) => call.method === "POST" && call.url.includes("/v1/folders"))).toHaveLength(0);
  });
});

describe("Chronological list row actions (⋮ menu)", () => {
  // The Recent list sorts by updated_at desc with store order breaking ties, so index 0 = s-ops-1.
  const openOpsMenu = () => fireEvent.click(screen.getAllByTestId("row-menu")[0]);

  it("rename / pin / archive / two-step delete all live behind the row's single kebab", async () => {
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
    ]);
    render(<Sidebar {...baseProps} />);
    await screen.findByText("incident watch"); // flat Recent list rendered

    // Rename: menu item → inline input → Enter commits.
    openOpsMenu();
    fireEvent.click(screen.getByTestId("row-menu-rename"));
    const input = screen.getByDisplayValue("incident watch");
    fireEvent.change(input, { target: { value: "war room" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(baseProps.onRenameSession).toHaveBeenCalledWith("s-ops-1", "war room");

    // Pin moved inside the menu (unpinned session → "Pin").
    openOpsMenu();
    fireEvent.click(screen.getByTestId("row-menu-pin"));
    expect(baseProps.onTogglePin).toHaveBeenCalledWith("s-ops-1", true);

    // Archive.
    openOpsMenu();
    fireEvent.click(screen.getByTestId("row-menu-archive"));
    expect(baseProps.onArchiveSession).toHaveBeenCalledWith("s-ops-1", true);

    // Delete is two-step: first click arms ("Delete?"), the second deletes.
    openOpsMenu();
    fireEvent.click(screen.getByTestId("row-menu-delete"));
    expect(baseProps.onDeleteSession).not.toHaveBeenCalled();
    expect(screen.getByTestId("row-menu-delete").textContent).toContain("Delete?");
    fireEvent.click(screen.getByTestId("row-menu-delete"));
    expect(baseProps.onDeleteSession).toHaveBeenCalledWith("s-ops-1");
  });

  it("the kebab and its menu never select the row; Escape closes the menu", async () => {
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
    ]);
    render(<Sidebar {...baseProps} />);
    await screen.findByText("incident watch");

    openOpsMenu();
    fireEvent.click(screen.getByTestId("row-menu-pin"));
    expect(baseProps.onSelectSession).not.toHaveBeenCalled();

    openOpsMenu();
    expect(screen.getByTestId("row-menu-rename")).toBeTruthy();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("row-menu-rename")).toBeNull();
  });

  it("moves a session to a folder through the secondary folder picker", async () => {
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
      {
        match: "/v1/folders",
        method: "GET",
        json: { folders: [{ id: "work", name: "Work", created: "2026-08-17T00:00:00Z" }] },
      },
    ]);
    render(<Sidebar {...baseProps} />);
    await screen.findByText("incident watch");

    openOpsMenu();
    fireEvent.click(screen.getByTestId("row-menu-folder"));
    fireEvent.click(within(screen.getByTestId("folder-picker-menu")).getByText("Work"));
    expect(baseProps.onSetSessionFolder).toHaveBeenCalledWith("s-ops-1", "work");
  });
});

describe("Sweep to archive", () => {
  it("requires a second click before archiving every session", async () => {
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
    ]);
    render(<Sidebar {...baseProps} />);
    const sweep = await screen.findByTestId("sweep-archive");

    fireEvent.click(sweep);
    expect(baseProps.onArchiveAllSessions).not.toHaveBeenCalled();
    expect(sweep.textContent).toContain("Archive all?");
    fireEvent.click(sweep);
    expect(baseProps.onArchiveAllSessions).toHaveBeenCalledTimes(1);
  });
});

describe("From Slack group (§31)", () => {
  const SLACK_SESSION: SessionInfo = {
    session_id: "s-slack-1",
    title: "#general — check the deploy?",
    workspace: "",
    agent: "cowork",
    model: "m",
    mode: "interactive",
    updated_at: "2026-07-13",
    messages: 2,
    origin: "slack",
    origin_label: "#general · T0AB",
  };

  it("mention-spawned sessions list chronologically in Recent with the platform icon (no band)", async () => {
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
    ]);
    render(<Sidebar {...baseProps} sessions={[...SESSIONS, SLACK_SESSION]} />);
    await screen.findByText("incident watch"); // flat Recent rendered

    // No collapsed band — the session sits directly in the Recent list, exactly once…
    expect(screen.queryByTestId("from-slack-toggle")).toBeNull();
    const row = await screen.findByText("#general — check the deploy?");
    expect(screen.getAllByText("#general — check the deploy?")).toHaveLength(1);

    // …wearing the Slack logo in the row's indicator cluster.
    const cluster = row.closest(".group");
    expect(cluster?.querySelector('[data-logo="slack"]')).toBeTruthy();
  });
});

describe("New-session split button", () => {
  it("keeps the folder menu in solo mode without repeating the sole persona", async () => {
    stubFetch([
      {
        match: "/v1/personas",
        method: "GET",
        json: { personas: [PERSONAS.personas[0], PERSONAS.personas[3]] }, // cowork + a disabled one
      },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
    ]);
    const { container } = render(<Sidebar {...baseProps} />);
    await screen.findByText("incident watch");

    // The primary still starts the sole enabled persona.
    fireEvent.click(container.querySelector(".newsplit-primary")!);
    expect(baseProps.onNewSession).toHaveBeenCalledWith("cowork");

    // The chevron remains useful for folder creation, without a redundant persona section.
    fireEvent.click(screen.getByLabelText("New session options"));
    expect(screen.queryByText("Start a session as")).toBeNull();
    fireEvent.click(await screen.findByText("New folder"));
    expect(await screen.findByPlaceholderText("Folder name")).toBeTruthy();
  });

  it("primary starts the last-used persona; the menu lists enabled personas + Manage personas…", async () => {
    localStorage.setItem("ocw.flag.personas", "1"); // Manage entry is launch-flagged off
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
    ]);
    const { container } = render(<Sidebar {...baseProps} />);
    await screen.findByLabelText("Group and filter conversations");

    // Primary action → a new session with the current (last-used) persona.
    fireEvent.click(container.querySelector(".newsplit-primary")!);
    expect(baseProps.onNewSession).toHaveBeenCalledWith("cowork");

    // ▾ opens the persona menu: enabled personas appear, the disabled one does not, plus a manage entry.
    fireEvent.click(screen.getByLabelText("New session options"));
    const menu = (await screen.findByText("Start a session as")).closest(".newsplit-menu") as HTMLElement;
    const w = within(menu);
    expect(w.getByText("Ops")).toBeTruthy();
    expect(w.getByText("Code")).toBeTruthy();
    expect(w.queryByText("Disabled One")).toBeNull();
    expect(w.getByText("Manage personas…")).toBeTruthy();

    // Selecting a persona starts a session as that persona.
    fireEvent.click(w.getByText("Ops"));
    expect(baseProps.onNewSession).toHaveBeenCalledWith("ops");

    // "Manage personas…" opens the persona management surface.
    fireEvent.click(screen.getByLabelText("New session options"));
    fireEvent.click(await screen.findByText("Manage personas…"));
    expect(baseProps.onManagePersonas).toHaveBeenCalled();
  });

  it("hides Manage personas… while the launch flag is off (the default)", async () => {
    localStorage.removeItem("ocw.flag.personas");
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
    ]);
    render(<Sidebar {...baseProps} />);
    await screen.findByLabelText("Group and filter conversations");
    fireEvent.click(screen.getByLabelText("New session options"));
    const menu = (await screen.findByText("Start a session as")).closest(".newsplit-menu") as HTMLElement;
    expect(within(menu).getByText("Ops")).toBeTruthy();
    expect(within(menu).queryByText("Manage personas…")).toBeNull();
  });
});
