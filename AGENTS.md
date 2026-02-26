# AGENTS

This repository must never use personal identity metadata in commits.

## Required Git Identity

- `user.name` must be `TARVORIX`
- `user.email` must be `TARVORIX@users.noreply.github.com`

## Rules

- Do not commit with machine-local or personal emails.
- Do not commit with real personal names.
- Never modify unrelated files. Only edit files required for the user-requested task.
- Never run repo-wide mutating commands that can touch unrelated files (for example `cargo fmt --all`, bulk formatters, or broad codemods) unless the user explicitly requests it.
- If unrelated files are changed accidentally, restore them immediately before continuing and report exactly what happened.
- Before committing, verify identity with:
  - `git config --get user.name`
  - `git config --get user.email`
- If identity is incorrect, set repo-local config before commit:
  - `git config user.name "TARVORIX"`
  - `git config user.email "TARVORIX@users.noreply.github.com"`

## Privacy

- Do not add personal paths, personal hostnames, personal emails, API keys, tokens, or secrets to tracked files.
