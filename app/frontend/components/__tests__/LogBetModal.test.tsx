import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LogBetModal } from "../LogBetModal";

const baseProps = {
  open: true,
  onClose: vi.fn(),
  homeTeam: "Villarreal",
  awayTeam: "Real Betis",
  statusLabel: "Today · Full Time",
  onSubmit: vi.fn(),
};

describe("LogBetModal", () => {
  it("renders nothing when closed", () => {
    const { container } = render(
      <LogBetModal {...baseProps} open={false} locked market="result_3way" selection="draw" odds={4.0} />
    );
    // Proves the dialog/backdrop DOM itself is gone (the early `if (!open)
    // return null`), not just that one piece of its text happens to be
    // absent -- a stray leftover element elsewhere wouldn't be caught by
    // checking "Log bet" text alone.
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(container).toBeEmptyDOMElement();
  });

  it("locked mode: shows Market/Pick/Odds as fixed text, not editable controls", () => {
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} />);

    expect(screen.getByText("Villarreal v Real Betis · via The Odds API")).toBeInTheDocument();
    expect(screen.getByText("3-Way Result")).toBeInTheDocument();
    expect(screen.getByText("Draw")).toBeInTheDocument();
    expect(screen.getByText("4.00")).toBeInTheDocument();
    expect(screen.queryByLabelText(/^market$/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/^pick$/i)).not.toBeInTheDocument();
  });

  it("editable mode: Market/Selection are real dropdowns, Odds is a real input", () => {
    render(
      <LogBetModal {...baseProps} locked={false} market="result_3way" selection="" odds={null} />
    );

    expect(screen.getByLabelText(/^market$/i).tagName).toBe("SELECT");
    expect(screen.getByLabelText(/^pick$/i).tagName).toBe("SELECT");
    expect(screen.getByLabelText(/^odds$/i)).toHaveAttribute("placeholder", "0.00");
  });

  it("editable mode: changing market resets the selection", async () => {
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked={false} market="result_3way" selection="home" odds={null} />);

    await user.selectOptions(screen.getByLabelText(/^market$/i), "btts");

    expect((screen.getByLabelText(/^pick$/i) as HTMLSelectElement).value).toBe("");
  });

  it("shows a live 'Returns $X.XX if it hits' as stake is typed, using the locked odds", async () => {
    // The dollar figure is deliberately its own styled <span> (a visual
    // callout, not plain text) inside the sentence -- getByText's default
    // string/regex matching only reads an element's direct text-node
    // children, so it can't see across that nested span. A function
    // matcher checking the whole paragraph's real (aggregated) textContent
    // is the standard RTL idiom for exactly this "styled substring inside
    // a sentence" shape.
    function returnsText(expected: string) {
      return (_: string, el: Element | null) => el?.tagName.toLowerCase() === "p" && el.textContent === expected;
    }

    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} />);

    expect(screen.getByText(returnsText("Returns $0.00 if it hits"))).toBeInTheDocument();

    await user.type(screen.getByLabelText(/^stake$/i), "10");

    expect(screen.getByText(returnsText("Returns $40.00 if it hits"))).toBeInTheDocument();
  });

  it("Cancel calls onClose without calling onSubmit", async () => {
    const onClose = vi.fn();
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onClose={onClose} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: /cancel/i }));

    expect(onClose).toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("the X button also calls onClose", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: /close/i }));

    expect(onClose).toHaveBeenCalled();
  });

  it("clicking the backdrop calls onClose", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    const { container } = render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onClose={onClose} />);

    await user.click(container.querySelector('[aria-hidden="true"]')!);

    expect(onClose).toHaveBeenCalled();
  });

  it("pressing Escape calls onClose", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onClose={onClose} />);

    await user.keyboard("{Escape}");

    expect(onClose).toHaveBeenCalled();
  });

  it("locks body scroll while open and restores it once closed", () => {
    document.body.style.overflow = "";
    const { rerender } = render(
      <LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} />
    );

    expect(document.body.style.overflow).toBe("hidden");

    rerender(<LogBetModal {...baseProps} open={false} locked market="result_3way" selection="draw" odds={4.0} />);

    expect(document.body.style.overflow).toBe("");
  });

  it("locked mode: a stake-only validation error, not the editable-mode 'fill in a pick' message", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(screen.getByText(/enter a stake greater than 0/i)).toBeInTheDocument();
    expect(screen.queryByText(/fill in a pick/i)).not.toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("Confirm bet calls onSubmit with the parsed fields (locked mode)", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText(/^stake$/i), "10");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(onSubmit).toHaveBeenCalledWith({ market: "result_3way", selection: "draw", odds: 4.0, stake: 10 });
  });

  it("Confirm bet calls onSubmit with the user-picked fields (editable mode)", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked={false} market="result_3way" selection="" odds={null} onSubmit={onSubmit} />);

    await user.selectOptions(screen.getByLabelText(/^market$/i), "btts");
    await user.selectOptions(screen.getByLabelText(/^pick$/i), "yes");
    await user.type(screen.getByLabelText(/^odds$/i), "1.9");
    await user.type(screen.getByLabelText(/^stake$/i), "5");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(onSubmit).toHaveBeenCalledWith({ market: "btts", selection: "yes", odds: 1.9, stake: 5 });
  });

  it("rejects submit with no stake / a zero odds / an unselected pick, without calling onSubmit", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked={false} market="result_3way" selection="" odds={null} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByText(/fill in/i)).toBeInTheDocument();
  });

  it("shows an inline error and keeps the modal open when onSubmit rejects", async () => {
    const onSubmit = vi.fn().mockRejectedValue(new Error("Could not log bet."));
    const user = userEvent.setup();
    render(<LogBetModal {...baseProps} locked market="result_3way" selection="draw" odds={4.0} onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText(/^stake$/i), "10");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(await screen.findByText("Could not log bet.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /confirm bet/i })).toBeInTheDocument();
  });
});
