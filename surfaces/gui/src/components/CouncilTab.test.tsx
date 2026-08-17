// Settings ▸ Council. The prompts are the product here, so the coverage that matters is
// that an edit reaches the server, and that Reset genuinely restores the shipped wording
// rather than freezing today's text as a user override.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { CouncilTab } from "./CouncilTab";
import * as api from "../api";

const CONFIG: api.CouncilConfig = {
  preset: "analysis",
  rounds: 2,
  research: true,
  depth: "standard",
  detail: "standard",
  panel: [],
  chair_model: "",
  roles: [
    { name: "Advocate", brief: "case for" },
    { name: "Skeptic", brief: "case against" },
  ],
  sources: [],
  prompts: {},
  skip_debate_on_agreement: true,
  max_tokens_per_run: 500_000,
  defaults: {
    analysis: { round1: "SHIPPED ROUND1", debate: "SHIPPED DEBATE", chair: "SHIPPED CHAIR" },
    decision: { round1: "D ROUND1", debate: "D DEBATE", chair: "D CHAIR" },
  },
  default_roles: [
    { name: "Advocate", brief: "case for" },
    { name: "Skeptic", brief: "case against" },
  ],
  source_kinds: ["folder", "file", "url", "search", "http", "mcp"],
  resolved_panel: [
    { model: "azure:gpt-5.6-sol", role: "Advocate" },
    { model: "xai:grok-4.5", role: "Skeptic" },
  ],
  resolved_chair: "azure:gpt-5.6-sol",
  depths: {
    quick: { rounds: 1, max_members: 3, research: false, label: "Quick", blurb: "Three models, once." },
    standard: { rounds: 2, max_members: 6, research: true, label: "Standard", blurb: "Answer then rebut." },
    deep: { rounds: 3, max_members: 8, research: true, label: "Deep", blurb: "Two rebuttal rounds." },
  },
  details: {
    brief: { label: "Brief", blurb: "The answer and what it turns on.", instruction: "under 200 words" },
    standard: { label: "Standard", blurb: "The answer and the reasoning.", instruction: "under 600 words" },
    full: { label: "Full", blurb: "Everything.", instruction: "as much as justified" },
  },
};

let saved: Partial<api.CouncilConfig>[];

beforeEach(() => {
  saved = [];
  vi.spyOn(api, "getCouncilConfig").mockResolvedValue(structuredClone(CONFIG));
  vi.spyOn(api, "setCouncilConfig").mockImplementation(async (patch) => {
    saved.push(patch);
    return { ...structuredClone(CONFIG), ...patch, ok: true };
  });
  vi.spyOn(api, "getCouncilRuns").mockResolvedValue([]);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("CouncilTab", () => {
  it("shows the panel that would actually run, with each member's lens", async () => {
    render(<CouncilTab />);
    const panel = await screen.findByTestId("resolved-panel");
    expect(panel.textContent).toContain("azure:gpt-5.6-sol");
    expect(panel.textContent).toContain("Advocate");
    expect(panel.textContent).toContain("Skeptic");
  });

  it("shows the shipped prompt when there is no override", async () => {
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "Prompts" }));
    const chair = (await screen.findByLabelText("Chair")) as HTMLTextAreaElement;
    expect(chair.value).toBe("SHIPPED CHAIR");
  });

  it("saves an edited prompt under the right preset and phase", async () => {
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "Prompts" }));
    const chair = await screen.findByLabelText("Chair");
    fireEvent.change(chair, { target: { value: "MY CHAIR" } });
    fireEvent.click(screen.getAllByText("Save")[2]);
    await waitFor(() => expect(saved.length).toBe(1));
    expect(saved[0].prompts).toEqual({ analysis: { chair: "MY CHAIR" } });
  });

  it("Reset clears the override rather than saving the shipped text as one", async () => {
    // Saving the default text back would pin it forever: the prompt would stop tracking
    // future improvements while looking untouched.
    vi.spyOn(api, "getCouncilConfig").mockResolvedValue({
      ...structuredClone(CONFIG),
      prompts: { analysis: { chair: "MY CHAIR" } },
    });
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "Prompts" }));
    expect(((await screen.findByLabelText("Chair")) as HTMLTextAreaElement).value).toBe("MY CHAIR");
    const resets = screen.getAllByText("Reset") as HTMLButtonElement[];
    // Only the edited prompt's Reset is live — the other two are already shipped text.
    expect(resets.map((b) => b.disabled)).toEqual([true, true, false]);
    fireEvent.click(resets[2]);
    await waitFor(() => expect(saved.length).toBe(1));
    expect(saved[0].prompts).toEqual({ analysis: { chair: "" } });
  });

  it("edits decision-mode prompts separately from analysis", async () => {
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "Prompts" }));
    fireEvent.change(screen.getByLabelText("Editing prompts for"), {
      target: { value: "decision" },
    });
    const chair = (await screen.findByLabelText("Chair")) as HTMLTextAreaElement;
    expect(chair.value).toBe("D CHAIR");
    fireEvent.change(chair, { target: { value: "MY DECISION CHAIR" } });
    fireEvent.click(screen.getAllByText("Save")[2]);
    await waitFor(() => expect(saved.length).toBe(1));
    expect(saved[0].prompts).toEqual({ decision: { chair: "MY DECISION CHAIR" } });
  });

  it("adds a source and saves it", async () => {
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "Sources" }));
    fireEvent.click(screen.getByText("Add source"));
    fireEvent.change(screen.getByLabelText("Source 1 target"), {
      target: { value: "/home/me/notes" },
    });
    fireEvent.click(screen.getByText("Save sources"));
    await waitFor(() => expect(saved.length).toBe(1));
    expect(saved[0].sources).toEqual([
      { kind: "folder", target: "/home/me/notes", label: "", options: {}, enabled: true },
    ]);
  });

  it("drops a source with no target instead of saving a blank one", async () => {
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "Sources" }));
    fireEvent.click(screen.getByText("Add source"));
    fireEvent.click(screen.getByText("Save sources"));
    await waitFor(() => expect(saved.length).toBe(1));
    expect(saved[0].sources).toEqual([]);
  });

  it("keeps the raw Options text while it is being typed", async () => {
    // Deriving the field's value from the parsed object throws away every keystroke that
    // makes the JSON temporarily invalid, so the field can only be edited by pasting.
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "Sources" }));
    fireEvent.click(screen.getByText("Add source"));
    const options = screen.getByLabelText("Source 1 options") as HTMLInputElement;

    fireEvent.change(options, { target: { value: '{"glob' } });
    expect(options.value).toBe('{"glob');
    expect(options.getAttribute("aria-invalid")).toBe("true");
    expect((screen.getByText("Save sources") as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(options, { target: { value: '{"glob": "**/*.md"}' } });
    expect(options.getAttribute("aria-invalid")).toBe("false");
    expect((screen.getByText("Save sources") as HTMLButtonElement).disabled).toBe(false);
  });

  it("saves the Options a valid draft parsed to", async () => {
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "Sources" }));
    fireEvent.click(screen.getByText("Add source"));
    fireEvent.change(screen.getByLabelText("Source 1 target"), { target: { value: "/docs" } });
    fireEvent.change(screen.getByLabelText("Source 1 options"), {
      target: { value: '{"glob": "**/*.md"}' },
    });
    fireEvent.click(screen.getByText("Save sources"));
    await waitFor(() => expect(saved.length).toBe(1));
    expect(saved[0].sources?.[0].options).toEqual({ glob: "**/*.md" });
  });

  it("reports a source test failure in place", async () => {
    vi.spyOn(api, "testCouncilSource").mockResolvedValue({
      ok: false,
      error: "FileNotFoundError: not a directory",
    });
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "Sources" }));
    fireEvent.click(screen.getByText("Add source"));
    fireEvent.click(screen.getByText("Test"));
    expect((await screen.findByRole("status")).textContent).toContain("not a directory");
  });

  it("saves a toggle immediately", async () => {
    render(<CouncilTab />);
    const toggle = await screen.findByLabelText(
      "Skip the debate when the opening round already agrees",
      { exact: false },
    );
    fireEvent.click(toggle);
    await waitFor(() => expect(saved.length).toBe(1));
    expect(saved[0].skip_debate_on_agreement).toBe(false);
  });

  it("explains the empty history state instead of showing a blank pane", async () => {
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "History" }));
    expect((await screen.findByText(/No councils yet/)).textContent).toContain("convene the council");
  });
});

describe("CouncilTab history", () => {
  it("never shows one run's content under another run's header", async () => {
    // One shared files map renders B's header over A's still-loaded markdown, and leaves
    // A's transcript on screen labelled as B if B's fetch fails.
    vi.spyOn(api, "getCouncilRuns").mockResolvedValue([
      { id: "run-a", updated_at: 2, files: ["finding.md"] },
      { id: "run-b", updated_at: 1, files: ["finding.md"] },
    ]);
    vi.spyOn(api, "getCouncilRun").mockImplementation(async (id) =>
      id === "run-a"
        ? { ok: true, id, files: { "finding.md": "FINDING A" } }
        : { ok: false, error: "no such council run" },
    );

    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "History" }));

    fireEvent.click(await screen.findByText("run-a"));
    expect((await screen.findByText(/FINDING A/)).textContent).toContain("FINDING A");

    fireEvent.click(screen.getByText("run-a")); // collapse
    fireEvent.click(screen.getByText("run-b"));
    expect(await screen.findByText("no such council run")).toBeTruthy();
    expect(screen.queryByText(/FINDING A/)).toBeNull();
  });

  it("reports a failed source test instead of hanging on Testing…", async () => {
    vi.spyOn(api, "testCouncilSource").mockRejectedValue(new Error("network down"));
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "Sources" }));
    fireEvent.click(screen.getByText("Add source"));
    fireEvent.click(screen.getByText("Test"));
    expect((await screen.findByRole("status")).textContent).toContain("network down");
  });
});

describe("CouncilTab cost controls", () => {
  it("saves a new round guard", async () => {
    render(<CouncilTab />);
    const field = await screen.findByLabelText("Stop adding rounds past");
    fireEvent.change(field, { target: { value: "120000" } });
    fireEvent.blur(field);
    await waitFor(() => expect(saved.length).toBe(1));
    expect(saved[0].max_tokens_per_run).toBe(120000);
  });

  it("clamps a negative value client-side; the server maps it back to the default", async () => {
    render(<CouncilTab />);
    const field = await screen.findByLabelText("Stop adding rounds past");
    fireEvent.change(field, { target: { value: "-5" } });
    fireEvent.blur(field);
    await waitFor(() => expect(saved.length).toBe(1));
    expect(saved[0].max_tokens_per_run).toBe(0);
  });
});

describe("CouncilTab history states", () => {
  it("distinguishes a failed run-list fetch from an empty history", async () => {
    // Swallowing the failure and falling through to "No councils yet" tells the user they
    // have no councils when the request actually failed — the one message guaranteed to
    // send them looking in the wrong place.
    vi.spyOn(api, "getCouncilRuns").mockRejectedValue(new Error("connection refused"));
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "History" }));
    expect((await screen.findByRole("status")).textContent).toContain("not the same");
    expect(screen.queryByText(/No councils yet/)).toBeNull();
  });

  it("still shows the empty state when the list genuinely comes back empty", async () => {
    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "History" }));
    expect(await screen.findByText(/No councils yet/)).toBeTruthy();
  });

  it("clears a run's previous error when a retry succeeds", async () => {
    vi.spyOn(api, "getCouncilRuns").mockResolvedValue([
      { id: "run-a", updated_at: 1, files: ["finding.md"] },
    ]);
    const get = vi
      .spyOn(api, "getCouncilRun")
      .mockResolvedValueOnce({ ok: false, error: "no such council run" })
      .mockResolvedValueOnce({ ok: true, id: "run-a", files: { "finding.md": "FINDING A" } });

    render(<CouncilTab />);
    fireEvent.click(await screen.findByRole("tab", { name: "History" }));

    fireEvent.click(await screen.findByText("run-a"));
    expect(await screen.findByText("no such council run")).toBeTruthy();

    fireEvent.click(screen.getByText("run-a")); // collapse
    fireEvent.click(screen.getByText("run-a")); // retry
    expect(await screen.findByText(/FINDING A/)).toBeTruthy();
    expect(screen.queryByText("no such council run")).toBeNull();
    expect(get).toHaveBeenCalledTimes(2);
  });
});
