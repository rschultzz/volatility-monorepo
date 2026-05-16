---
description: Write end-of-session journal entry to the Obsidian vault
---

Wrap up this session by writing a journal entry to the Obsidian vault, following the protocol in `CLAUDE.md`.

The vault path contains a space — quote it in any shell commands.

Steps:

1. Read `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/_templates/session-template.md` to get the current template.
2. Decide a slug — 2 to 4 hyphenated words capturing the session's primary topic (e.g. `gex-clustering-bug`, `long-strategy-tuning`, `databento-backfill-validation`).
3. Save a new file at `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/sessions/YYYY-MM-DD-<slug>.md` using today's date.
4. Fill in the frontmatter completely. Use `[]` for empty lists rather than leaving fields blank. Be specific in `files_touched` and `components`.
5. Body sections: Goal, What changed, Decisions, Open questions, Next session.
6. If a strategy parameter changed during this session, also update the relevant file under `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/strategies/`.
7. If we resolved an item in `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/open-questions/`, set its frontmatter `status: resolved` and link the session note.
8. If we made an architectural decision, write a short ADR-style note under `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/decisions/`.
9. Print the absolute path of the new session note when finished.
