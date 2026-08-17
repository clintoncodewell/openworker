import { describe, expect, it } from "vitest";
import { zoomsOnTitlebarDoubleClick } from "./tauri";

// Regression: the first version of this guard required target === currentTarget, so the
// only place a double-click did NOT zoom was the visible title — which is where anyone
// aiming at "double-click the titlebar" actually clicks.
describe("zoomsOnTitlebarDoubleClick", () => {
  const titlebar = () => {
    const bar = document.createElement("div");
    bar.className = "main-title";
    bar.innerHTML = `
      <span class="main-title-text">Some session</span>
      <span class="title-sub">gpt-5 · interactive</span>
      <button type="button" data-testid="session-settings">settings</button>
      <a href="#docs">docs</a>
    `;
    document.body.appendChild(bar);
    return bar;
  };

  it("zooms when the titlebar itself is double-clicked", () => {
    expect(zoomsOnTitlebarDoubleClick(titlebar())).toBe(true);
  });

  it("zooms when the title text is double-clicked", () => {
    const bar = titlebar();
    expect(zoomsOnTitlebarDoubleClick(bar.querySelector(".main-title-text"))).toBe(true);
    expect(zoomsOnTitlebarDoubleClick(bar.querySelector(".title-sub"))).toBe(true);
  });

  it("does not zoom on a control, or on anything inside one", () => {
    const bar = titlebar();
    expect(zoomsOnTitlebarDoubleClick(bar.querySelector("button"))).toBe(false);
    expect(zoomsOnTitlebarDoubleClick(bar.querySelector("a"))).toBe(false);
  });

  it("does not throw on a null or non-element target", () => {
    expect(zoomsOnTitlebarDoubleClick(null)).toBe(true);
    expect(zoomsOnTitlebarDoubleClick(document.createTextNode("x") as unknown as EventTarget)).toBe(
      true,
    );
  });
});
