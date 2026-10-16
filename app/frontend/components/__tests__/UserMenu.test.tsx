import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { UserMenu } from "../UserMenu";

const mockUseSession = vi.fn();
const mockSignOut = vi.fn();
vi.mock("next-auth/react", () => ({
  useSession: () => mockUseSession(),
  signOut: (...args: unknown[]) => mockSignOut(...args),
}));

const mockIsPushSubscribed = vi.fn();
const mockSubscribeToPush = vi.fn();
const mockUnsubscribeFromPush = vi.fn();
vi.mock("@/lib/push", () => ({
  isPushSubscribed: (...args: unknown[]) => mockIsPushSubscribed(...args),
  subscribeToPush: (...args: unknown[]) => mockSubscribeToPush(...args),
  unsubscribeFromPush: (...args: unknown[]) => mockUnsubscribeFromPush(...args),
}));

describe("UserMenu", () => {
  beforeEach(() => {
    mockUseSession.mockReset();
    mockSignOut.mockReset();
    mockIsPushSubscribed.mockReset().mockResolvedValue(false);
    mockSubscribeToPush.mockReset().mockResolvedValue(undefined);
    mockUnsubscribeFromPush.mockReset().mockResolvedValue(undefined);
  });

  it("renders a Sign in link when unauthenticated", () => {
    mockUseSession.mockReturnValue({ data: null, status: "unauthenticated" });
    render(<UserMenu />);
    expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute("href", "/login");
  });

  it("renders nothing while the session is loading", () => {
    mockUseSession.mockReturnValue({ data: null, status: "loading" });
    const { container } = render(<UserMenu />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the signed-in email and a working Sign out button", async () => {
    mockUseSession.mockReturnValue({
      data: { user: { email: "fpai.deploy@gmail.com" } },
      status: "authenticated",
    });
    render(<UserMenu />);
    expect(screen.getByText("fpai.deploy@gmail.com")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /sign out/i }));
    expect(mockSignOut).toHaveBeenCalledWith({ callbackUrl: "/" });
  });

  it("shows a notifications toggle for a signed-in user when push is supported", async () => {
    Object.defineProperty(window.navigator, "serviceWorker", { value: {}, configurable: true });
    Object.defineProperty(window, "PushManager", { value: function () {}, configurable: true });
    mockUseSession.mockReturnValue({
      data: { user: { email: "fpai.deploy@gmail.com" } },
      status: "authenticated",
    });
    render(<UserMenu />);
    expect(await screen.findByRole("button", { name: /turn on notifications/i })).toBeInTheDocument();
  });

  it("subscribes when the toggle is clicked while off", async () => {
    Object.defineProperty(window.navigator, "serviceWorker", { value: {}, configurable: true });
    Object.defineProperty(window, "PushManager", { value: function () {}, configurable: true });
    mockUseSession.mockReturnValue({
      data: { user: { email: "fpai.deploy@gmail.com" } },
      status: "authenticated",
    });
    render(<UserMenu />);
    const toggle = await screen.findByRole("button", { name: /turn on notifications/i });
    await userEvent.click(toggle);
    expect(mockSubscribeToPush).toHaveBeenCalled();
    expect(await screen.findByRole("button", { name: /turn off notifications/i })).toBeInTheDocument();
  });

  it("unsubscribes when the toggle is clicked while on", async () => {
    Object.defineProperty(window.navigator, "serviceWorker", { value: {}, configurable: true });
    Object.defineProperty(window, "PushManager", { value: function () {}, configurable: true });
    mockIsPushSubscribed.mockReset().mockResolvedValue(true);
    mockUseSession.mockReturnValue({
      data: { user: { email: "fpai.deploy@gmail.com" } },
      status: "authenticated",
    });
    render(<UserMenu />);
    const toggle = await screen.findByRole("button", { name: /turn off notifications/i });
    await userEvent.click(toggle);
    expect(mockUnsubscribeFromPush).toHaveBeenCalled();
    expect(await screen.findByRole("button", { name: /turn on notifications/i })).toBeInTheDocument();
  });

  it("hides the toggle when the browser has no PushManager support", () => {
    Object.defineProperty(window.navigator, "serviceWorker", { value: undefined, configurable: true });
    Object.defineProperty(window, "PushManager", { value: undefined, configurable: true });
    mockUseSession.mockReturnValue({
      data: { user: { email: "fpai.deploy@gmail.com" } },
      status: "authenticated",
    });
    render(<UserMenu />);
    expect(screen.queryByText(/notifications/i)).not.toBeInTheDocument();
  });
});
