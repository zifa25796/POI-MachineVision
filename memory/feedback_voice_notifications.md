---
name: voice-notifications
description: Use VoiceLine MCP to play spoken notifications at end of every turn
metadata:
  type: feedback
---

At the end of every response / turn, call the VoiceLine MCP tool:

  mcp__voice-line__voice_speak

with a short message (under 10 words) summarising what just happened.

**Why:** User wants audio alerts instead of watching the terminal.
[[voice-notifications]] in user memory is the canonical copy; this is a project-level mirror.
