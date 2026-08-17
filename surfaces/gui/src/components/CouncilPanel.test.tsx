// The live debate panel. A council blocks for minutes and says nothing while it works, so
// what this shows — who is thinking, where they stand, what they posted — is the whole
// difference between "running" and "hung".
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { CouncilPanel } from "./CouncilPanel";
import type { CouncilLive } from "../api";

afterEach(cleanup);

const LIVE: CouncilLive = {
  run: "20260817-x",
  question: "Should we ship on Friday?",
  status: "round 1 done",
  round: 1,
  rounds: 2,
  queries: ["friday deploy risk", "deployment freeze policy"],
  panel: [
    { model: "azure:gpt-5.6-sol", role: "Advocate" },
    { model: "xai:grok-4.6", role: "Skeptic" },
    { model: "claude-code:claude-opus-5", role: "Pragmatist" },
  ],
  stances: [
    { model: "azure:gpt-5.6-sol", role: "Advocate", stance: "ship it", confidence: "confident", error: "" },
    { model: "xai:grok-4.6", role: "Skeptic", stance: "wait", confidence: "unsure", error: "" },
  ],
  notes: [{ model: "azure:gpt-5.6-sol", role: "Advocate", note: "the queue drains by 4pm", round: "1" }],
};

describe("CouncilPanel", () => {
  it("shows where each member stands, and who has not answered yet", () => {
    render(<CouncilPanel live={LIVE} />);
    expect(screen.getByText("ship it · confident")).toBeTruthy();
    expect(screen.getByText("wait · unsure")).toBeTruthy();
    // The third member has not reported — that is the state the panel exists to show.
    expect(screen.getByText("thinking…")).toBeTruthy();
  });

  it("drops the vendor prefix — the column is narrow and the prefix is noise", () => {
    render(<CouncilPanel live={LIVE} />);
    // It appears twice — once as a member, once as the author of a scratchpad note.
    expect(screen.getAllByText("gpt-5.6-sol").length).toBeGreaterThan(0);
    expect(screen.queryByText("azure:gpt-5.6-sol")).toBeNull();
  });

  it("reports confidence in words, never as a number", () => {
    render(<CouncilPanel live={LIVE} />);
    expect(screen.queryByText(/0\.\d/)).toBeNull();
  });

  it("shows the round it is on while it runs, and stops counting when done", () => {
    const { rerender } = render(<CouncilPanel live={LIVE} />);
    expect(screen.getByText("round 1 of 2")).toBeTruthy();

    rerender(<CouncilPanel live={{ ...LIVE, status: "done" }} />);
    expect(screen.getByText("Finding ready")).toBeTruthy();
    expect(screen.queryByText(/round 1 of 2/)).toBeNull();
  });

  it("shows the searches that were actually run", () => {
    render(<CouncilPanel live={LIVE} />);
    // The fix for the research bug is only checkable if the queries are visible.
    expect(screen.getByText(/friday deploy risk · deployment freeze policy/)).toBeTruthy();
  });

  it("shows the shared scratchpad as it fills", () => {
    render(<CouncilPanel live={LIVE} />);
    expect(screen.getByText(/the queue drains by 4pm/)).toBeTruthy();
  });

  it("names the members that failed once the run is done", () => {
    render(
      <CouncilPanel
        live={{
          ...LIVE,
          status: "done",
          stances: [
            ...(LIVE.stances || []),
            { model: "claude-code:claude-opus-5", role: "Pragmatist", stance: "", confidence: "", error: "not on PATH" },
          ],
        }}
      />,
    );
    expect(screen.getByText(/1 member did not answer: claude-opus-5/)).toBeTruthy();
  });

  it("surfaces the run's own notes, so a silent degradation is not silent", () => {
    render(
      <CouncilPanel
        live={{
          ...LIVE,
          status: "done",
          report: { notes: ["Web research returned nothing — the panel argued from memory."] },
        }}
      />,
    );
    expect(screen.getByText(/argued from memory/)).toBeTruthy();
  });

  it("renders a panel that has not reported anything yet", () => {
    render(<CouncilPanel live={{ run: "x", status: "researching", panel: [], rounds: 2 }} />);
    expect(screen.getByTestId("council-live")).toBeTruthy();
    expect(screen.getByText("researching")).toBeTruthy();
  });
});
