import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { PullToRefreshIndicator } from "../PullToRefreshIndicator";

describe("PullToRefreshIndicator", () => {
  it("renders nothing when idle (pullDistance 0, not locked)", () => {
    const { container } = render(
      <PullToRefreshIndicator pullDistance={0} locked={false} dragging={false} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("grows to match pullDistance while dragging", () => {
    const { container } = render(
      <PullToRefreshIndicator pullDistance={40} locked={false} dragging={true} />
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.style.height).toBe("40px");
  });

  it("locks at the threshold height and spins once locked", () => {
    const { container } = render(
      <PullToRefreshIndicator pullDistance={70} locked={true} dragging={false} />
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.style.height).toBe("70px");
    expect(container.querySelector(".animate-spin")).toBeInTheDocument();
  });
});
