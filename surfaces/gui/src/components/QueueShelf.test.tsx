import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueueShelf, type QueueItem } from "./QueueShelf";

afterEach(cleanup);

const items: QueueItem[] = [
  {
    id: "q-1",
    text: "Review the report",
    attachments: [{ kind: "text", name: "report.txt" }],
    not_before: 0,
  },
  { id: "q-2", text: "Then summarize it" },
];

function props(overrides = {}) {
  return {
    items,
    paused: false,
    running: true,
    onSteerNow: vi.fn(),
    onRemove: vi.fn(),
    onEdit: vi.fn(),
    onReorder: vi.fn(),
    onResume: vi.fn(),
    ...overrides,
  };
}

describe("QueueShelf", () => {
  it("shows scheduling and attachment state", () => {
    render(<QueueShelf {...props()} />);
    expect(screen.getByText(/1 scheduled/)).toBeTruthy();
    expect(screen.getByText("1 attachment")).toBeTruthy();
    expect(screen.getByLabelText("Pending prompts").parentElement?.className).toContain("shrink-0");
  });

  it("guards repeated actions until an authoritative snapshot arrives", () => {
    const p = props();
    const { rerender } = render(<QueueShelf {...p} />);
    const steer = screen.getAllByText("Steer now")[0] as HTMLButtonElement;
    fireEvent.click(steer);
    fireEvent.click(steer);
    expect(p.onSteerNow).toHaveBeenCalledTimes(1);
    expect(steer.disabled).toBe(true);

    rerender(<QueueShelf {...p} items={[...items]} />);
    expect((screen.getAllByText("Steer now")[0] as HTMLButtonElement).disabled).toBe(false);
  });

  it("never sends an empty edit", () => {
    const p = props();
    render(<QueueShelf {...p} />);
    fireEvent.click(screen.getByLabelText("Edit item 1"));
    const editor = screen.getByDisplayValue("Review the report");
    fireEvent.change(editor, { target: { value: "   " } });
    fireEvent.click(screen.getByText("Save"));
    expect(p.onEdit).not.toHaveBeenCalled();
  });
});
