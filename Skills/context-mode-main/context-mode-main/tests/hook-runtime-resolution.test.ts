/**
 * Hook runtime resolution — issue #738.
 *
 * Auto-detect bun for hook command emission to cut ~40-60ms cold-start per
 * tool-call. Falls back to node when bun is missing, too old, or fails the
 * version probe. Resolution is cached at module load.
 *
 * Design constraints (locked with Mert, 2026-05-31):
 *   - NO env var, NO opt-out flag. User's escape is uninstall bun.
 *   - bun ≥ 1.0 required (older versions had ESM bugs that broke hooks).
 *   - Silent fallback on any failure — never block hook execution.
 *   - Module-load probe with cached result; not per-call.
 *   - `buildNodeCommand` semantics UNCHANGED (kept for openclaw doctor /
 *     upgrade hints which must stay on node — better-sqlite3 ABI, #543).
 *   - New `buildHookRuntimeCommand` wraps `buildNodeCommand` shape but
 *     swaps in the resolved JS runtime when bun is available.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

// node:fs / node:child_process mocks below pin POSIX-shape bunFallbackPaths()
// candidates (~/.bun/bin/bun). Windows resolves to %USERPROFILE%\.bun\bin\bun.exe
// AND additional %LOCALAPPDATA%\Programs\bun\* candidates; faithfully mocking
// that generator from the test side is brittle and has already gone red twice
// chasing one-off mismatches. The production code path itself is Windows-safe
// (bunCommand() handles the .exe suffix + %LOCALAPPDATA% trap from #506); we
// guard those invariants in tests/runtime.test.ts which uses the real fs.
// Skip the POSIX-mock-only cases on Windows so CI stops getting blocked by
// test-infra fragility while still exercising the same logic on Ubuntu+macOS.
const itPosix = process.platform === "win32" ? test.skip : test;

describe("resolveHookRuntime — auto-detect bun ≥1.0, fall back to node (#738)", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.resetModules();
    vi.doUnmock("node:child_process");
    vi.doUnmock("node:fs");
  });

  itPosix("returns bun path + isBun=true when bun ≥1.0 is available", async () => {
    // bunCommand() resolves to either:
    //   1) the first existing path in bunFallbackPaths() (~/.bun/bin/bun on Unix), OR
    //   2) the literal "bun" string when commandExists("bun") succeeds.
    // We exercise branch (1) by stubbing HOME to a known directory and
    // claiming only "$HOME/.bun/bin/bun" exists.
    const fakeHome = "/fake/home/for-738-test";
    const fakeBun = `${fakeHome}/.bun/bin/bun`;
    const originalHome = process.env.HOME;
    process.env.HOME = fakeHome;
    try {
      const execSync = vi.fn((cmd: string) => {
        if (/^command -v\s+bun$/.test(cmd)) return `${fakeBun}\n`;
        return "";
      });
      const execFileSync = vi.fn((cmd: string, args: string[]) => {
        if (cmd === fakeBun && args[0] === "--version") {
          return Buffer.from("1.1.0\n");
        }
        throw new Error(`unexpected execFile: ${cmd} ${args.join(" ")}`);
      });
      vi.doMock("node:child_process", () => ({ execSync, execFileSync }));
      vi.doMock("node:fs", async () => {
        const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
        return {
          ...actual,
          existsSync: (p: string | URL) => String(p) === fakeBun,
        };
      });

      const { resolveHookRuntime, resetHookRuntimeCache } = await import("../src/runtime.js");
      resetHookRuntimeCache();
      const r = resolveHookRuntime();
      expect(r.isBun).toBe(true);
      expect(r.path).toBe(fakeBun);
    } finally {
      if (originalHome === undefined) delete process.env.HOME;
      else process.env.HOME = originalHome;
    }
  });

  test("returns node + isBun=false when bun is not installed", async () => {
    const execSync = vi.fn((cmd: string) => {
      if (/^command -v\s+bun$/.test(cmd)) throw new Error("not found");
      if (/^where\s+bun$/.test(cmd)) throw new Error("not found");
      return "";
    });
    const execFileSync = vi.fn(() => Buffer.from(""));
    vi.doMock("node:child_process", () => ({ execSync, execFileSync }));
    vi.doMock("node:fs", async () => {
      const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
      return { ...actual, existsSync: () => false };
    });

    const { resolveHookRuntime, resetHookRuntimeCache } = await import("../src/runtime.js");
    resetHookRuntimeCache();
    const r = resolveHookRuntime();
    expect(r.isBun).toBe(false);
    expect(r.path).toBe(process.execPath);
  });

  itPosix("returns node + isBun=false when bun version < 1.0", async () => {
    const fakeHome = "/fake/home/for-738-test";
    const fakeBun = `${fakeHome}/.bun/bin/bun`;
    const originalHome = process.env.HOME;
    process.env.HOME = fakeHome;
    try {
      const execSync = vi.fn((cmd: string) => {
        if (/^command -v\s+bun$/.test(cmd)) return `${fakeBun}\n`;
        return "";
      });
      const execFileSync = vi.fn(() => Buffer.from("0.8.1\n"));
      vi.doMock("node:child_process", () => ({ execSync, execFileSync }));
      vi.doMock("node:fs", async () => {
        const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
        return {
          ...actual,
          existsSync: (p: string | URL) => String(p) === fakeBun,
        };
      });

      const { resolveHookRuntime, resetHookRuntimeCache } = await import("../src/runtime.js");
      resetHookRuntimeCache();
      const r = resolveHookRuntime();
      expect(r.isBun).toBe(false);
      expect(r.path).toBe(process.execPath);
    } finally {
      if (originalHome === undefined) delete process.env.HOME;
      else process.env.HOME = originalHome;
    }
  });

  itPosix("returns node + isBun=false when bun version probe crashes", async () => {
    const fakeHome = "/fake/home/for-738-test";
    const fakeBun = `${fakeHome}/.bun/bin/bun`;
    const originalHome = process.env.HOME;
    process.env.HOME = fakeHome;
    try {
      const execSync = vi.fn((cmd: string) => {
        if (/^command -v\s+bun$/.test(cmd)) return `${fakeBun}\n`;
        return "";
      });
      const execFileSync = vi.fn(() => {
        throw new Error("segfault");
      });
      vi.doMock("node:child_process", () => ({ execSync, execFileSync }));
      vi.doMock("node:fs", async () => {
        const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
        return {
          ...actual,
          existsSync: (p: string | URL) => String(p) === fakeBun,
        };
      });

      const { resolveHookRuntime, resetHookRuntimeCache } = await import("../src/runtime.js");
      resetHookRuntimeCache();
      const r = resolveHookRuntime();
      expect(r.isBun).toBe(false);
      expect(r.path).toBe(process.execPath);
    } finally {
      if (originalHome === undefined) delete process.env.HOME;
      else process.env.HOME = originalHome;
    }
  });

  itPosix("returns node + isBun=false when bun reports unparseable version string", async () => {
    const fakeHome = "/fake/home/for-738-test";
    const fakeBun = `${fakeHome}/.bun/bin/bun`;
    const originalHome = process.env.HOME;
    process.env.HOME = fakeHome;
    try {
      const execSync = vi.fn((cmd: string) => {
        if (/^command -v\s+bun$/.test(cmd)) return `${fakeBun}\n`;
        return "";
      });
      const execFileSync = vi.fn(() => Buffer.from("not-a-version\n"));
      vi.doMock("node:child_process", () => ({ execSync, execFileSync }));
      vi.doMock("node:fs", async () => {
        const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
        return {
          ...actual,
          existsSync: (p: string | URL) => String(p) === fakeBun,
        };
      });

      const { resolveHookRuntime, resetHookRuntimeCache } = await import("../src/runtime.js");
      resetHookRuntimeCache();
      const r = resolveHookRuntime();
      expect(r.isBun).toBe(false);
      expect(r.path).toBe(process.execPath);
    } finally {
      if (originalHome === undefined) delete process.env.HOME;
      else process.env.HOME = originalHome;
    }
  });

  test("caches result across calls (only probes once)", async () => {
    const execSync = vi.fn((cmd: string) => {
      if (/^command -v\s+bun$/.test(cmd)) throw new Error("nope");
      return "";
    });
    const execFileSync = vi.fn(() => Buffer.from(""));
    vi.doMock("node:child_process", () => ({ execSync, execFileSync }));
    vi.doMock("node:fs", async () => {
      const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
      return { ...actual, existsSync: () => false };
    });

    const { resolveHookRuntime, resetHookRuntimeCache } = await import("../src/runtime.js");
    resetHookRuntimeCache();
    const r1 = resolveHookRuntime();
    const probeCallCount = execSync.mock.calls.length + execFileSync.mock.calls.length;
    const r2 = resolveHookRuntime();
    const r3 = resolveHookRuntime();
    const afterCount = execSync.mock.calls.length + execFileSync.mock.calls.length;
    expect(afterCount).toBe(probeCallCount); // no new probes
    expect(r2).toEqual(r1);
    expect(r3).toEqual(r1);
  });
});

describe("buildHookRuntimeCommand — emits bun when available, node otherwise (#738)", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.resetModules();
    vi.doUnmock("node:child_process");
    vi.doUnmock("node:fs");
  });

  itPosix("emits bun path when bun ≥1.0 is resolved", async () => {
    const fakeHome = "/fake/home/for-738-test";
    const fakeBun = `${fakeHome}/.bun/bin/bun`;
    const originalHome = process.env.HOME;
    process.env.HOME = fakeHome;
    try {
      const execSync = vi.fn((cmd: string) => {
        if (/^command -v\s+bun$/.test(cmd)) return `${fakeBun}\n`;
        return "";
      });
      const execFileSync = vi.fn(() => Buffer.from("1.2.0\n"));
      vi.doMock("node:child_process", () => ({ execSync, execFileSync }));
      vi.doMock("node:fs", async () => {
        const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
        return {
          ...actual,
          existsSync: (p: string | URL) => String(p) === fakeBun,
        };
      });

      const { resetHookRuntimeCache } = await import("../src/runtime.js");
      resetHookRuntimeCache();
      const { buildHookRuntimeCommand } = await import("../src/adapters/types.js");
      const cmd = buildHookRuntimeCommand("/plugin/hooks/pretooluse.mjs");
      expect(cmd).toBe(`"${fakeBun}" "/plugin/hooks/pretooluse.mjs"`);
    } finally {
      if (originalHome === undefined) delete process.env.HOME;
      else process.env.HOME = originalHome;
    }
  });

  test("emits node (process.execPath) when bun is unavailable", async () => {
    const execSync = vi.fn((cmd: string) => {
      if (/^command -v\s+bun$/.test(cmd)) throw new Error("not found");
      if (/^where\s+bun$/.test(cmd)) throw new Error("not found");
      return "";
    });
    const execFileSync = vi.fn(() => Buffer.from(""));
    vi.doMock("node:child_process", () => ({ execSync, execFileSync }));
    vi.doMock("node:fs", async () => {
      const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
      return { ...actual, existsSync: () => false };
    });

    const { resetHookRuntimeCache } = await import("../src/runtime.js");
    resetHookRuntimeCache();
    const { buildHookRuntimeCommand } = await import("../src/adapters/types.js");
    const cmd = buildHookRuntimeCommand("/plugin/hooks/pretooluse.mjs");
    const nodePath = process.execPath.replace(/\\/g, "/");
    expect(cmd).toBe(`"${nodePath}" "/plugin/hooks/pretooluse.mjs"`);
  });

  itPosix("output is parseable by parseNodeCommand (round-trip invariant)", async () => {
    const fakeHome = "/fake/home/for-738-test";
    const fakeBun = `${fakeHome}/.bun/bin/bun`;
    const originalHome = process.env.HOME;
    process.env.HOME = fakeHome;
    try {
      const execSync = vi.fn((cmd: string) => {
        if (/^command -v\s+bun$/.test(cmd)) return `${fakeBun}\n`;
        return "";
      });
      const execFileSync = vi.fn(() => Buffer.from("1.1.0\n"));
      vi.doMock("node:child_process", () => ({ execSync, execFileSync }));
      vi.doMock("node:fs", async () => {
        const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
        return {
          ...actual,
          existsSync: (p: string | URL) => String(p) === fakeBun,
        };
      });

      const { resetHookRuntimeCache } = await import("../src/runtime.js");
      resetHookRuntimeCache();
      const { buildHookRuntimeCommand, parseNodeCommand } = await import("../src/adapters/types.js");
      const cmd = buildHookRuntimeCommand("/plugin/hooks/pretooluse.mjs");
      const parsed = parseNodeCommand(cmd);
      expect(parsed).not.toBeNull();
      expect(parsed!.scriptPath).toBe("/plugin/hooks/pretooluse.mjs");
      expect(parsed!.nodePath).toBe(fakeBun);
    } finally {
      if (originalHome === undefined) delete process.env.HOME;
      else process.env.HOME = originalHome;
    }
  });

  itPosix("buildNodeCommand semantics UNCHANGED — always returns process.execPath", async () => {
    // Regression guard: openclaw doctor/upgrade hints embed buildNodeCommand
    // output as user-facing copy-paste suggestions. They MUST stay on node
    // because the CLI needs better-sqlite3 (#543 bun ABI mismatch).
    const fakeHome = "/fake/home/for-738-test";
    const fakeBun = `${fakeHome}/.bun/bin/bun`;
    const originalHome = process.env.HOME;
    process.env.HOME = fakeHome;
    try {
      const execSync = vi.fn((cmd: string) => {
        if (/^command -v\s+bun$/.test(cmd)) return `${fakeBun}\n`;
        return "";
      });
      const execFileSync = vi.fn(() => Buffer.from("1.1.0\n"));
      vi.doMock("node:child_process", () => ({ execSync, execFileSync }));
      vi.doMock("node:fs", async () => {
        const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
        return {
          ...actual,
          existsSync: (p: string | URL) => String(p) === fakeBun,
        };
      });

      const { resetHookRuntimeCache } = await import("../src/runtime.js");
      resetHookRuntimeCache();
      const { buildNodeCommand } = await import("../src/adapters/types.js");
      const cmd = buildNodeCommand("/cli.bundle.mjs");
      // Even with bun available, buildNodeCommand stays on node.
      expect(cmd).not.toContain(fakeBun);
      expect(cmd).toContain(process.execPath.replace(/\\/g, "/"));
    } finally {
      if (originalHome === undefined) delete process.env.HOME;
      else process.env.HOME = originalHome;
    }
  });
});
