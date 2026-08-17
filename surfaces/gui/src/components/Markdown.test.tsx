import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { Markdown, OPEN_ARTIFACT_EVENT } from "./Markdown";

afterEach(cleanup);

// §34 (UX-016): [Title](artifact:path) renders as a chip that opens the artifact viewer via
// a window event; ordinary links keep the open-externally treatment.
describe("Markdown artifact links", () => {
  it("renders an artifact: link as a chip and dispatches the open event with the path", () => {
    const seen: string[] = [];
    const listener = (e: Event) => seen.push((e as CustomEvent).detail.path);
    window.addEventListener(OPEN_ARTIFACT_EVENT, listener);

    render(<Markdown text="Done — [Semiconductor dashboard](artifact:reports/semi.html)" />);
    const chip = screen.getByTestId("artifact-chip");
    expect(chip.textContent).toContain("Semiconductor dashboard");
    expect(chip.textContent).toContain("semi.html"); // filename shown under the title
    fireEvent.click(chip);
    expect(seen).toEqual(["reports/semi.html"]);

    window.removeEventListener(OPEN_ARTIFACT_EVENT, listener);
  });

  it("ordinary links stay external and never become chips", () => {
    const { container } = render(<Markdown text="see [the docs](https://example.com)" />);
    expect(screen.queryByTestId("artifact-chip")).toBeNull();
    const a = container.querySelector("a")!;
    expect(a.getAttribute("target")).toBe("_blank");
    expect(a.getAttribute("href")).toBe("https://example.com");
  });

  it("chip title falls back to the filename when the link text is empty", () => {
    vi.spyOn(window, "dispatchEvent");
    render(<Markdown text="[](artifact:out/report.pdf)" />);
    expect(screen.getByTestId("artifact-chip").textContent).toContain("report.pdf");
  });
});

// Source links in a council finding were dead in the shipped app: `target="_blank"` opens a
// tab in the dev build and does nothing in the desktop webview, which has no window to open.
describe("Markdown external links", () => {
  afterEach(() => delete (globalThis as any).__TAURI__);

  it("hands an http link to the desktop opener instead of a dead _blank", () => {
    const openUrl = vi.fn(async () => {});
    (globalThis as any).__TAURI__ = { opener: { openUrl } };

    render(<Markdown text="See [MSCI](https://www.msci.com/wealth) for the data." />);
    fireEvent.click(screen.getByText("MSCI"));
    expect(openUrl).toHaveBeenCalledWith("https://www.msci.com/wealth");
  });

  it("falls back to window.open outside the desktop app", () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null);
    render(<Markdown text="See [MSCI](https://www.msci.com/wealth)." />);
    fireEvent.click(screen.getByText("MSCI"));
    expect(open).toHaveBeenCalled();
    open.mockRestore();
  });

  it("leaves a modified click alone, so open-in-new-window still works", () => {
    const openUrl = vi.fn(async () => {});
    (globalThis as any).__TAURI__ = { opener: { openUrl } };
    render(<Markdown text="[MSCI](https://www.msci.com/wealth)" />);
    fireEvent.click(screen.getByText("MSCI"), { metaKey: true });
    expect(openUrl).not.toHaveBeenCalled();
  });

  it("still autolinks a bare URL the panel pasted", () => {
    render(<Markdown text="Source: https://www.msci.com/wealth" />);
    expect(screen.getByText("https://www.msci.com/wealth").tagName).toBe("A");
  });
});
