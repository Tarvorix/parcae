# AGENTS

This repository must never use personal identity metadata in commits.

## Required Git Identity

- `user.name` must be `TARVORIX`
- `user.email` must be `TARVORIX@users.noreply.github.com`

## Rules

- Do not commit with machine-local or personal emails.
- Do not commit with real personal names.
- Before committing, verify identity with:
  - `git config --get user.name`
  - `git config --get user.email`
- If identity is incorrect, set repo-local config before commit:
  - `git config user.name "TARVORIX"`
  - `git config user.email "TARVORIX@users.noreply.github.com"`

## Privacy

- Do not add personal paths, personal hostnames, personal emails, API keys, tokens, or secrets to tracked files.
