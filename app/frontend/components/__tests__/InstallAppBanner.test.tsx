import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, beforeEach, vi } from "vitest";
import { InstallAppBanner } from "../InstallAppBanner";

function mockMatchMedia(matches: boolean) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    configurable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  });
}

describe("InstallAppBanner", () => {
  beforeEach(() => {
    localStorage.clear();
    mockMatchMedia(false);
    Object.defineProperty(window.navigator, "standalone", { value: false, configurable: true });
  });

  it("shows iOS Add-to-Home-Screen instructions on an iOS user agent", () => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"
    );
    render(<InstallAppBanner />);
    expect(screen.getByText(/add to home screen/i)).toBeInTheDocument();
  });

  it("does not render when already running standalone", () => {
    mockMatchMedia(true);
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"
    );
    render(<InstallAppBanner />);
    expect(screen.queryByText(/install oddsey/i)).not.toBeInTheDocument();
  });

  it("does not render when previously dismissed", () => {
    localStorage.setItem("oddsey-install-banner-dismissed", "1");
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"
    );
    render(<InstallAppBanner />);
    expect(screen.queryByText(/add to home screen/i)).not.toBeInTheDocument();
  });

  it("shows an Install button on Android and calls prompt() on click", async () => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/120.0 Mobile Safari/537.36"
    );
    render(<InstallAppBanner />);
    const promptFn = vi.fn();
    const event = new Event("beforeinstallprompt", { cancelable: true }) as Event & { prompt: () => void };
    event.prompt = promptFn;
    window.dispatchEvent(event);
    const button = await screen.findByRole("button", { name: /^install$/i });
    fireEvent.click(button);
    expect(promptFn).toHaveBeenCalled();
  });
});
