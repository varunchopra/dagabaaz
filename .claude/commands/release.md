---
description: Bump version, commit, tag, and push to trigger PyPI publish
allowed-tools: Bash(git *), Bash(uv *), Bash(grep *), Read, Edit
---

Release version $ARGUMENTS to PyPI.

Pre-flight checks (stop and report if any fail):
1. Verify working tree is clean: `git status --short` must produce no output
2. Verify current branch is main: `git branch --show-current` must output `main`
3. Verify local main is up to date: `git fetch origin main && git diff --quiet main origin/main` -- if it fails, tell user to pull first
4. Verify tag doesn't exist: `git tag -l "v$ARGUMENTS"` must produce no output
5. Verify `$ARGUMENTS` looks like a valid semver (e.g. 0.2.0, 1.0.0)
6. Verify `pyproject.toml` has a static `version` field under `[project]`, not listed in `dynamic`

Steps:
1. Update the `version` field in `pyproject.toml` to `$ARGUMENTS`
2. Run `uv run ruff check src/ tests/` to verify linting passes -- if it fails, run `git checkout pyproject.toml` to revert and stop
3. Run `uv run pytest` to verify tests pass -- if they fail, run `git checkout pyproject.toml` to revert and stop
4. Run `uv lock` to update the lockfile
5. Stage only `pyproject.toml` and `uv.lock`: `git add pyproject.toml uv.lock`
6. Commit with message `release: v$ARGUMENTS` (no Co-Authored-By trailers)
7. Create annotated git tag: `git tag -a "v$ARGUMENTS" -m "Release v$ARGUMENTS"`
8. Push commit and tag separately: `git push origin main && git push origin "v$ARGUMENTS"`

If tests fail, revert pyproject.toml with `git checkout pyproject.toml` and stop. Do not commit or tag.
