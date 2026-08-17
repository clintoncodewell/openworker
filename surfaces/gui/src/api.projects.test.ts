// The list route wraps its result — `{projects: [...]}` — while every other project route
// returns the object bare. The component tests mock this module wholesale, so they return
// whatever shape they are told and cannot see that asymmetry at all. This exercises the
// real client against the real server shape, which is the gap that let the first build ship
// a Projects screen that called .map on an object and showed nothing.
import { afterEach, describe, expect, it, vi } from "vitest";
import { getProjects } from "./api";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

const answer = (body: unknown) => {
  globalThis.fetch = vi.fn(async () => ({ json: async () => body })) as never;
};

describe("getProjects", () => {
  it("unwraps the shape the server actually sends", async () => {
    answer({ projects: [{ id: "p1", name: "One", session_count: 0, updated_at: 0 }] });
    expect((await getProjects()).map((p) => p.name)).toEqual(["One"]);
  });

  it("accepts a bare array too, so a server change degrades rather than throwing", async () => {
    answer([{ id: "p1", name: "One", session_count: 0, updated_at: 0 }]);
    expect((await getProjects()).map((p) => p.name)).toEqual(["One"]);
  });

  it("returns nothing rather than crashing on an unexpected body", async () => {
    answer({ unexpected: true });
    expect(await getProjects()).toEqual([]);
  });
});
