#!/bin/bash
# Wrapper invoked by the launchd agent. Ensures the LM Studio server is up
# (if installed) before running an incremental index.
set -u

# launchd jobs don't inherit the user's interactive PATH.
export PATH="$HOME/.local/bin:$HOME/.lmstudio/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

# Best-effort: start the LM Studio server. Idempotent — already-running is fine.
# Skipped silently if `lms` is unavailable (e.g., Ollama- or Gemini-only setup).
if command -v lms >/dev/null 2>&1; then
    lms server start >/dev/null 2>&1 || true
fi

exec obsidian-semantic index
