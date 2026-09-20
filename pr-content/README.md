# LLaDA2-Uni PR content collaboration

This directory is the shared editing area for the titles and descriptions of
seven existing pull requests in `sgl-project/sglang-omni`. It belongs only to
the `llada2/PR-content` branch of `kunchengit/sglang-omni`.

## Pull requests

| Pull request | Title | Description |
| --- | --- | --- |
| [#1486](https://github.com/sgl-project/sglang-omni/pull/1486) | [title.txt](1486/title.txt) | [body.md](1486/body.md) |
| [#1487](https://github.com/sgl-project/sglang-omni/pull/1487) | [title.txt](1487/title.txt) | [body.md](1487/body.md) |
| [#1490](https://github.com/sgl-project/sglang-omni/pull/1490) | [title.txt](1490/title.txt) | [body.md](1490/body.md) |
| [#1499](https://github.com/sgl-project/sglang-omni/pull/1499) | [title.txt](1499/title.txt) | [body.md](1499/body.md) |
| [#1500](https://github.com/sgl-project/sglang-omni/pull/1500) | [title.txt](1500/title.txt) | [body.md](1500/body.md) |
| [#1501](https://github.com/sgl-project/sglang-omni/pull/1501) | [title.txt](1501/title.txt) | [body.md](1501/body.md) |
| [#1502](https://github.com/sgl-project/sglang-omni/pull/1502) | [title.txt](1502/title.txt) | [body.md](1502/body.md) |

## Editing and publishing

1. Edit the relevant `title.txt` and/or `body.md` on this collaboration branch.
   The title file contains only the PR title; the body file contains only the
   Markdown description to publish.
2. Commit the wording changes here for collaborator review. Keep code changes
   on their existing PR branches.
3. Before publishing, compare the files with the current upstream PR. If someone
   has edited the PR on GitHub, reconcile those edits before replacing its text.
4. The PR author or someone with the required upstream permissions publishes
   the agreed title and description, then reads them back to verify the result.

Editing these files does **not** automatically update GitHub. No PR-content
publishing script or workflow is configured here, and no credentials or
additional upstream permissions are included.

Do not open an upstream PR for this collaboration branch or merge it into
`main` or the feature branches. Only public, publication-ready PR content
belongs here; do not add credentials or private development material.

## Initial snapshot

`snapshot.json` records the source URLs, source branch names, status, and capture
time for the initial export. GitHub's `updated_at` is a general PR update time,
not necessarily the time its title or description was last edited.

All seven PRs were open drafts when captured. These status fields are a snapshot,
not a live status report.

The initial title and description files preserve the GitHub text exactly,
including line endings, blank lines, and whether a final newline is present.
The scoped `.gitattributes` prevents Git from normalizing these files on checkout
or staging. Do not automatically reformat the initial export. In particular,
the repository's `end-of-file-fixer` hook would add a newline to descriptions
that originally have none; `.gitattributes` does not prevent hook or editor edits.

The title and description files are the editable working copies. The snapshot
metadata describes their initial source and does not need to change for every
wording edit.
