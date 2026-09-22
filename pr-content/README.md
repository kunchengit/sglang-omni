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

### Implementation mapping (2026-09-22)

The five descriptions below were refreshed against the published implementation
branches, rather than the initial PR export. Titles remain unchanged.

| PR | Implementation branch in `kunchengit/sglang-omni` | Reviewed head | Immediate base |
| --- | --- | --- | --- |
| #1499 | `llada2/native-image-generation` | `f2e6fbbdb4a5` | `llada2/thinker-fix` (`cf71f2fbbb02`) |
| #1500 | `llada2/thinking-image-generation` | `5c4e5c615bf0` | #1499 |
| #1502 | `llada2/interleaved-image-generation` | `11932284ec72` | #1500 |
| #1486 | `llada2/thinker-tp` | `f520bba6c78d` | #1502 |
| #1501 | `llada2/decoder-sp` | `b6d0a7d82231` | #1502 |

The dependency chain is thinker correctness -> native image -> thinking image ->
interleaved generation, followed by separate thinker-TP and decoder-SP branches.
The interleaved branch includes shared relay work; the decoder-SP branch includes
generic stage SP work. The older #1487/#1490 descriptions are left untouched, but
their overlapping scope must be reconciled before publication/merge.

The #1501 head branch, `llada2/decoder-sp`, has been synchronized with
`pipeline/stage-sp` at `b6d0a7d82231`. Both branches now contain the same
implementation, including the Omni-only sequence-order correction. The text
stored here still needs to be published separately by the PR author.

Validation sections distinguish historical GPU evidence from checks after the
latest stack synchronization. Historical precomputed-token edit scores are not
claims about the raw-image path; the public `.pt` input has been removed.

### Publishing workflow

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
