// Council mode: a per-chat toggle that convenes the whole panel on every message.
// It prepends "Council:" rather than threading a hidden flag through the send path — the
// transcript then shows the message the user actually sent, and they can edit or override it.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { Composer, withCouncil } from "./Composer";

const props = (extra: Partial<Parameters<typeof Composer>[0]> = {}) => ({
  sessionId: "test-session",
  mode: "interactive",
  model: "gpt-5.6-sol",
  running: false,
  connected: true,
  onSend: vi.fn(),
  onInterrupt: vi.fn(),
  onModeChange: vi.fn(),
  onModelChange: vi.fn(),
  ...extra,
});

const type = (text: string) => {
  const box = screen.getByRole("textbox");
  fireEvent.change(box, { target: { value: text } });
  fireEvent.keyDown(box, { key: "Enter" });
};

beforeEach(() => localStorage.clear());
afterEach(cleanup);

describe("withCouncil", () => {
  it("prefixes the question when the toggle is on", () => {
    expect(withCouncil("should we ship?", true)).toBe("Council: should we ship?");
  });

  it("leaves the text alone when it is off", () => {
    expect(withCouncil("should we ship?", false)).toBe("should we ship?");
  });

  it("never doubles a prefix the user typed themselves", () => {
    expect(withCouncil("Council: ship?", true)).toBe("Council: ship?");
    expect(withCouncil("council: ship?", true)).toBe("council: ship?");
  });

  it("leaves an empty draft empty — a bare prefix is not a question", () => {
    expect(withCouncil("   ", true)).toBe("");
  });
});

describe("Composer council toggle", () => {
  it("is off until asked for", () => {
    const onSend = vi.fn();
    render(<Composer {...props({ onSend })} />);
    type("ship?");
    expect(onSend).toHaveBeenCalledWith("ship?", []);
  });

  it("sends every message to the panel once on", () => {
    const onSend = vi.fn();
    render(<Composer {...props({ onSend })} />);
    fireEvent.click(screen.getByTestId("council-toggle"));
    type("ship?");
    expect(onSend).toHaveBeenCalledWith("Council: ship?", []);
  });

  it("stays on for the chat, and off for a different one", () => {
    const { unmount } = render(<Composer {...props({ resetKey: "chat-a" })} />);
    fireEvent.click(screen.getByTestId("council-toggle"));
    expect(screen.getByTestId("council-toggle").getAttribute("aria-pressed")).toBe("true");
    unmount();
    cleanup();

    // Reopening the same chat keeps it; a different chat starts clean.
    render(<Composer {...props({ resetKey: "chat-a" })} />);
    expect(screen.getByTestId("council-toggle").getAttribute("aria-pressed")).toBe("true");
    cleanup();

    render(<Composer {...props({ resetKey: "chat-b" })} />);
    expect(screen.getByTestId("council-toggle").getAttribute("aria-pressed")).toBe("false");
  });

  it("does not convene a second panel when steering a running turn", () => {
    const onSend = vi.fn();
    const onSteer = vi.fn();
    render(<Composer {...props({ onSend, onSteer, running: true })} />);
    fireEvent.click(screen.getByTestId("council-toggle"));
    const box = screen.getByRole("textbox");
    fireEvent.change(box, { target: { value: "focus on cost" } });
    fireEvent.keyDown(box, { key: "Enter", metaKey: true });
    expect(onSteer).toHaveBeenCalledWith("focus on cost", []);
  });
});

describe("Composer council toggle — the on state is actually visible", () => {
  it("changes its label and carries a class the stylesheet can win with", () => {
    // The first version used Tailwind's `bg-accent`, which loses to `.pill.chip` on
    // specificity. The button rendered identically in both states and the only way to
    // find out was to send a message.
    render(<Composer {...props()} />);
    const btn = screen.getByTestId("council-toggle");
    expect(btn.className).not.toContain("is-on");
    expect(btn.textContent).toBe("Council");

    fireEvent.click(btn);
    expect(btn.className).toContain("is-on");
    expect(btn.textContent).toBe("Council on");
  });

  it("says which way the click goes, in the tooltip", () => {
    render(<Composer {...props()} />);
    const btn = screen.getByTestId("council-toggle");
    expect(btn.getAttribute("title")).toContain("Click to turn on");
    fireEvent.click(btn);
    expect(btn.getAttribute("title")).toContain("Click to turn off");
  });

  it("does not rely on colour alone to show state", () => {
    render(<Composer {...props()} />);
    const btn = screen.getByTestId("council-toggle");
    // A dot for sighted users in greyscale, aria-pressed for a screen reader.
    expect(btn.querySelector(".pill-dot")).toBeTruthy();
    expect(btn.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(btn);
    expect(btn.getAttribute("aria-pressed")).toBe("true");
  });
});
