import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ProjectsView } from "./ProjectsView";
import * as api from "../api";
import type { SessionInfo } from "../types";

const markdown = (purpose: string, where = "The work is moving. More detail follows.") => `## Purpose
${purpose}

## Where it stands
${where}

## Decisions
- Use the local service.

## Open threads
- [ ] Confirm the launch date.`;

function project(
  id: string,
  name: string,
  count = 0,
  where = "The work is moving. More detail follows.",
): api.Project {
  return {
    id,
    name,
    created: "2026-08-01T10:00:00Z",
    session_ids: Array.from({ length: count }, (_, index) => `${id}-session-${index}`),
    instructions: "Keep answers concise.",
    project_md: markdown("Ship a dependable release.", where),
    sessions: [],
    files: [],
    updated_at: `2026-08-${String(10 + count).padStart(2, "0")}T10:00:00Z`,
  };
}

const RECENT_SESSION: SessionInfo = {
  session_id: "recent-1",
  title: "Launch planning",
  workspace: "/work/launch",
  agent: "cowork",
  model: "test",
  mode: "interactive",
  updated_at: "2026-08-17T10:00:00Z",
  messages: 4,
};

let records: api.Project[];

beforeEach(() => {
  records = [];
  vi.spyOn(api, "getProjects").mockImplementation(async () =>
    records.map(({ id, name, session_ids, updated_at }) => ({
      id,
      name,
      session_count: session_ids.length,
      updated_at,
    })),
  );
  vi.spyOn(api, "getProject").mockImplementation(async (id) => records.find((item) => item.id === id)!);
  vi.spyOn(api, "createProject").mockImplementation(async ({ name, purpose = "" }) => ({
    ...project("created", name),
    project_md: markdown(purpose, ""),
  }));
  vi.spyOn(api, "updateProject").mockImplementation(async (id, patch) => {
    const current = records.find((item) => item.id === id)!;
    return {
      ...current,
      name: patch.name ?? current.name,
      instructions: patch.instructions ?? current.instructions,
      project_md:
        patch.purpose === undefined
          ? current.project_md
          : markdown(patch.purpose, "The work is moving. More detail follows."),
    };
  });
  vi.spyOn(api, "refreshProject").mockImplementation(async (id) => records.find((item) => item.id === id)!);
  vi.spyOn(api, "deleteProject").mockImplementation(async (id) => ({ ok: true, id }));
  vi.spyOn(api, "addProjectSession").mockImplementation(async (id, sessionId) => {
    const current = records.find((item) => item.id === id)!;
    return {
      ...current,
      session_ids: [...current.session_ids, sessionId],
      sessions: [...current.sessions, RECENT_SESSION],
    };
  });
  vi.spyOn(api, "announceProjectsChanged").mockImplementation(() => true);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const renderProjects = (
  recentSessions: SessionInfo[] = [],
  onNewConversation = vi.fn(),
) =>
  render(
    <ProjectsView
      recentSessions={recentSessions}
      onOpenSession={vi.fn()}
      onNewConversation={onNewConversation}
    />,
  );

async function openFirstProject() {
  renderProjects([RECENT_SESSION]);
  fireEvent.click(await screen.findByText(records[0].name));
  await screen.findByRole("heading", { name: records[0].name });
}

describe("ProjectsView", () => {
  it("shows the empty state copy and its create button", async () => {
    renderProjects();
    expect(
      await screen.findByText(
        "A project keeps one topic together: its conversations, its files, and a brief that keeps itself up to date. Start one and every chat you put in it feeds the same picture.",
      ),
    ).toBeTruthy();
    expect(screen.getAllByRole("button", { name: "New project" })).toHaveLength(2);
  });

  it("renders summaries and pluralised conversation footers", async () => {
    records = [
      project("zero", "Zero", 0),
      project("one", "One", 1),
      project("many", "Many", 3),
    ];
    renderProjects();

    expect(await screen.findAllByText("The work is moving.")).toHaveLength(3);
    expect(screen.getByText(/No conversations yet · updated/)).toBeTruthy();
    expect(screen.getByText(/1 conversation · updated/)).toBeTruthy();
    expect(screen.getByText(/3 conversations · updated/)).toBeTruthy();
  });

  it("shows the no-summary message for an empty brief section", async () => {
    records = [project("empty", "Fresh project", 0, "")];
    renderProjects();
    expect(await screen.findByText("No summary yet — it fills in after the first conversation.")).toBeTruthy();
  });

  it("opens Purpose and the three self-maintained brief sections", async () => {
    records = [project("alpha", "Alpha")];
    await openFirstProject();

    expect((screen.getByLabelText("Purpose") as HTMLTextAreaElement).value).toBe(
      "Ship a dependable release.",
    );
    expect(screen.getByRole("heading", { name: "Where it stands" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Decisions" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Open threads" })).toBeTruthy();
    expect(screen.getByText("Use the local service.")).toBeTruthy();
    expect(screen.getByText("Confirm the launch date.")).toBeTruthy();
  });

  it("saves Purpose on blur but never saves an empty name", async () => {
    records = [project("alpha", "Alpha")];
    await openFirstProject();

    fireEvent.click(screen.getByRole("heading", { name: "Alpha" }));
    const name = screen.getByLabelText("Project name");
    fireEvent.change(name, { target: { value: "   " } });
    fireEvent.blur(name);
    expect(api.updateProject).not.toHaveBeenCalled();

    const purpose = screen.getByLabelText("Purpose");
    fireEvent.change(purpose, { target: { value: "Prepare the public launch." } });
    fireEvent.blur(purpose);
    await waitFor(() =>
      expect(api.updateProject).toHaveBeenCalledWith("alpha", {
        purpose: "Prepare the public launch.",
      }),
    );
  });

  it("refreshes the brief and shows a pending state", async () => {
    records = [project("alpha", "Alpha")];
    let resolveRefresh!: (value: api.ProjectMutationResult) => void;
    vi.mocked(api.refreshProject).mockImplementation(
      () => new Promise((resolve) => { resolveRefresh = resolve; }),
    );
    await openFirstProject();

    fireEvent.click(screen.getByRole("button", { name: "Refresh now" }));
    expect(screen.getByRole("button", { name: "Refreshing…" })).toBeTruthy();
    expect(api.refreshProject).toHaveBeenCalledWith("alpha");
    resolveRefresh(records[0]);
    await screen.findByRole("button", { name: "Refresh now" });
  });

  it("requires two clicks to delete a project", async () => {
    records = [project("alpha", "Alpha")];
    await openFirstProject();

    fireEvent.click(screen.getByRole("button", { name: "Delete project" }));
    expect(api.deleteProject).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Really delete?" }));
    await waitFor(() => expect(api.deleteProject).toHaveBeenCalledWith("alpha"));
  });

  it("adds the chosen recent conversation", async () => {
    records = [project("alpha", "Alpha")];
    await openFirstProject();

    fireEvent.change(screen.getByLabelText("Recent conversation"), {
      target: { value: "recent-1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() =>
      expect(api.addProjectSession).toHaveBeenCalledWith("alpha", "recent-1"),
    );
  });

  it("keeps rendering and shows the server error when attach returns an error payload", async () => {
    records = [project("alpha", "Alpha")];
    vi.mocked(api.addProjectSession).mockResolvedValue({
      ok: false,
      error: "no such project",
    });
    await openFirstProject();

    fireEvent.change(screen.getByLabelText("Recent conversation"), {
      target: { value: "recent-1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));

    expect((await screen.findByRole("alert")).textContent).toContain("no such project");
    expect(screen.getByRole("heading", { name: "Alpha" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "New conversation" })).toBeTruthy();
  });

  it("starts a new project conversation through the supplied app callback", async () => {
    records = [project("alpha", "Alpha")];
    const onNewConversation = vi.fn(() => new Promise<void>(() => {}));
    renderProjects([RECENT_SESSION], onNewConversation);
    fireEvent.click(await screen.findByText("Alpha"));

    fireEvent.click(screen.getByRole("button", { name: "New conversation" }));
    expect(onNewConversation).toHaveBeenCalledWith("alpha");
    expect(
      (screen.getByRole("button", { name: "Starting…" }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});
