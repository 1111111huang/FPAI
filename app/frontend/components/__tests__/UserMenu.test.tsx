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

describe("UserMenu", () => {
  beforeEach(() => {
    mockUseSession.mockReset();
    mockSignOut.mockReset();
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
});
