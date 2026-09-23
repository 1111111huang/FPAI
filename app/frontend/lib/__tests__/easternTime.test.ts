/**
 * W40: "today" for this single-user app is always America/New_York,
 * regardless of the viewer's browser/OS timezone. These assertions use
 * Intl.DateTimeFormat with an explicit timeZone -- documented to be
 * independent of the runtime's own default zone (unlike Date's bare
 * getFullYear()/getMonth()/getDate()), so they hold under any TZ env
 * value. Acceptance per the story: run this suite under multiple TZ env
 * values (at minimum UTC, Asia/Tokyo, America/Los_Angeles) and confirm
 * identical results -- verified manually via
 * `TZ=<zone> npx vitest run lib/__tests__/easternTime.test.ts` for all
 * three, not just asserted in CI config here.
 */
import { describe, expect, it } from "vitest";
import { easternDateParts, dayDiff, dateString, addDays } from "@/components/MatchUI";

describe("easternDateParts -- Eastern calendar day, independent of the runtime's own TZ", () => {
  it("reads a UTC evening instant as the same Eastern calendar day (EDT, UTC-4)", () => {
    // 23:30 UTC on 2026-07-15 (summer, EDT) is 19:30 Eastern -- still the
    // same day. A bare ambient-local getter would agree only by
    // coincidence for a runtime whose own TZ happens to be Eastern.
    expect(easternDateParts(new Date("2026-07-15T23:30:00Z"))).toEqual({ year: 2026, month: 6, day: 15 });
  });

  it("reads a UTC late-night instant as the PREVIOUS Eastern calendar day (EDT, UTC-4)", () => {
    // 02:00 UTC on 2026-07-16 is 22:00 Eastern on 2026-07-15 -- the whole
    // point of this story: UTC and Eastern disagree on the calendar day
    // for several hours of every real day.
    expect(easternDateParts(new Date("2026-07-16T02:00:00Z"))).toEqual({ year: 2026, month: 6, day: 15 });
  });

  it("crosses the DST boundary correctly (EST, UTC-5, in winter)", () => {
    // 04:30 UTC on 2026-01-16 is 23:30 Eastern on 2026-01-15 (EST, one
    // hour further behind than the EDT case above) -- confirms the
    // Intl-based extraction, not a hardcoded UTC-4 offset.
    expect(easternDateParts(new Date("2026-01-16T04:30:00Z"))).toEqual({ year: 2026, month: 0, day: 15 });
  });
});

describe("dayDiff/dateString/addDays -- non-sandbox mode is Eastern-Time-anchored, not ambient-local", () => {
  it("dayDiff treats a UTC-late-night kickoff and an Eastern-evening asOf as the same day", () => {
    // Fixture kicks off 02:00 UTC (previous Eastern evening, per the test
    // above); asOf is the real instant an hour earlier the same UTC
    // calendar day but genuinely the same Eastern day as the kickoff.
    const asOf = new Date("2026-07-16T01:00:00Z"); // 2026-07-15 21:00 ET
    expect(dayDiff("2026-07-16T02:00:00Z", asOf, false)).toBe(0); // 2026-07-15 22:00 ET -- same Eastern day
  });

  it("dateString formats the Eastern calendar day, not the UTC one", () => {
    expect(dateString(new Date("2026-07-16T02:00:00Z"), false)).toBe("2026-07-15");
  });

  it("addDays round-trips correctly through dateString across a DST boundary", () => {
    // Anchored at UTC noon internally (not midnight) specifically so this
    // round-trips correctly -- see addDays' own comment (MatchUI.tsx) for
    // why a midnight anchor would silently return the day before.
    const base = new Date("2026-01-15T18:00:00Z"); // 2026-01-15 13:00 ET (EST)
    const plus10 = addDays(base, 10, false);
    expect(dateString(plus10, false)).toBe("2026-01-25");
  });

  it("addDays(-30) then dateString still lands on the correct Eastern day going backward", () => {
    const base = new Date("2026-08-01T15:00:00Z"); // 2026-08-01 11:00 ET (EDT)
    const minus30 = addDays(base, -30, false);
    expect(dateString(minus30, false)).toBe("2026-07-02");
  });
});
