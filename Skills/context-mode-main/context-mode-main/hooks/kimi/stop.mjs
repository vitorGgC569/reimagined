#!/usr/bin/env node
import "./platform.mjs";
import "../suppress-stderr.mjs";
import "../ensure-deps.mjs";
/**
 * Kimi Code CLI Stop hook — record turn-end state for continuity.
 *
 * Stop fires at the END OF EACH ASSISTANT TURN, not at session close.
 * Kimi Code emits a distinct `SessionEnd` event for genuine session
 * shutdown (refs/platforms/kimi-code/.../session/index.ts:192,502 —
 * `triggerSessionEnd('exit')`); the matching `hooks/kimi/sessionend.mjs`
 * owns the `session_end` SessionDB row. Writing `session_end` here would
 * have produced one such row per turn.
 *   Cross-reference: refs/platforms/kimi-cli/src/kimi_cli/hooks/events.py:
 *     99-114 — `session_start` and `session_end` are distinct emitters.
 */

import { readStdin, parseStdin, getSessionId, getSessionDBPath, getInputProjectDir, KIMI_OPTS } from "../session-helpers.mjs";
import { createSessionLoaders } from "../session-loaders.mjs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HOOK_DIR = dirname(fileURLToPath(import.meta.url));
const { loadSessionDB } = createSessionLoaders(HOOK_DIR);
const OPTS = KIMI_OPTS;

try {
  const raw = await readStdin();
  const input = parseStdin(raw);
  const projectDir = getInputProjectDir(input, OPTS);

  const { SessionDB } = await loadSessionDB();
  const dbPath = getSessionDBPath(OPTS, projectDir);
  const db = new SessionDB({ dbPath });
  const sessionId = getSessionId(input, OPTS);

  db.ensureSession(sessionId, projectDir);
  // SessionEvent contract (src/types.ts:33-47) requires `type`, `category`,
  // `data`, `priority`. SessionDB.insertEvent hashes `event.data` for the
  // dedup key — passing `undefined` throws inside the wrapping try and the
  // row silently never lands. Encode the turn snapshot into `data` so the
  // hash is stable and the row actually persists.
  const payload = {
    stop_hook_active: input.stop_hook_active ?? false,
    last_assistant_message: typeof input.last_assistant_message === "string"
      ? input.last_assistant_message.slice(0, 2000)
      : null,
  };
  db.insertEvent(sessionId, {
    type: "turn_end",
    category: "session",
    data: JSON.stringify(payload),
    priority: 1,
  }, "Stop");

  db.close();
} catch {
  // Kimi Code hooks must not block the session.
}

process.stdout.write("{}\n");
