import { describe, expect, it } from "vitest";
import { formatMetric, formatMs, formatUsd, methodLabel } from "./format";

describe("format", () => {
  it("formats metrics to 3 decimals and missing as a dash", () => {
    expect(formatMetric(0.8123)).toBe("0.812");
    expect(formatMetric(1)).toBe("1.000");
    expect(formatMetric(null)).toBe("—");
    expect(formatMetric(undefined)).toBe("—");
  });
  it("formats milliseconds", () => {
    expect(formatMs(3.14159)).toBe("3.1 ms");
    expect(formatMs(1534)).toBe("1.53 s");
  });
  it("formats cost", () => {
    expect(formatUsd(0)).toBe("$0");
    expect(formatUsd(0.0042)).toBe("$0.0042");
    expect(formatUsd(12.5)).toBe("$12.50");
  });
  it("labels methods", () => {
    expect(methodLabel("hybrid_rerank")).toBe("Hybrid + rerank");
    expect(methodLabel("bm25")).toBe("BM25");
  });
});

describe("formatBytes", () => {
  it("uses B, KB and MB", async () => {
    const { formatBytes } = await import("./format");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});
