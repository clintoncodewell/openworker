import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import * as api from "../api";
import { SettingsView } from "./SettingsView";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("Settings usage", () => {
  it("shows a loading state, quota meters, headroom, and manual usage link", async () => {
    let resolveUsage!: (value: api.UsageResponse) => void;
    const request = new Promise<api.UsageResponse>((resolve) => {
      resolveUsage = resolve;
    });
    vi.spyOn(api, "getUsage").mockReturnValue(request);

    render(<SettingsView initialTab="usage" />);
    expect(screen.getByLabelText("Loading usage")).toBeTruthy();

    resolveUsage({
      providers: [
        {
          id: "chatgpt",
          title: "ChatGPT subscription",
          status: "ok",
          kind: "quota_window",
          label: "Subscription window (approximate, unofficial endpoint)",
          windows: [
            { id: "primary", label: "Primary window", used_percent: 32, remaining_percent: 68 },
            { id: "secondary", label: "Weekly window", used_percent: 71, remaining_percent: 29 },
          ],
        },
        {
          id: "openai",
          title: "OpenAI",
          status: "ok",
          kind: "rate_limit_headroom",
          label: "Current throughput headroom — not a billing figure",
          metrics: [{ id: "requests", label: "Requests", remaining: "421", limit: "500", reset: "12s" }],
        },
        {
          id: "gemini",
          title: "Gemini (Google)",
          status: "ok",
          kind: "status_only",
          message: "Configured",
          link: "https://aistudio.google.com/usage",
        },
      ],
    });

    expect(await screen.findByText("68% left")).toBeTruthy();
    expect(screen.getByText("29% left")).toBeTruthy();
    expect(screen.getByText("421")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Check in Google AI Studio" }).getAttribute("href")).toBe(
      "https://aistudio.google.com/usage",
    );
  });

  it("refreshes on demand", async () => {
    const getUsage = vi.spyOn(api, "getUsage").mockResolvedValue({ providers: [] });
    render(<SettingsView initialTab="usage" />);

    await screen.findByText("Connect an account or API provider in Models to see usage here.");
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));

    expect(getUsage).toHaveBeenCalledTimes(2);
  });
});
