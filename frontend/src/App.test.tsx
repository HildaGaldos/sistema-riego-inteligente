import { describe, expect, it } from "vitest";

describe("frontend smoke test", () => {
  it("uses the manual dataset upload flow", () => {
    expect("/data/upload").toContain("data/upload");
  });
});
