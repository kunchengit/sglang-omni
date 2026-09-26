# LLaDA2-Uni PR content collaboration

This directory contains editable titles and descriptions for the active
LLaDA2-Uni contribution stack and roadmap. It belongs to the
`llada2/PR-content` branch of `kunchengit/sglang-omni`; editing these files does
not change GitHub automatically.

## Active pull requests

| Item | Pull request | Title | Description | Immediate prerequisite |
| --- | --- | --- | --- | --- |
| A | [#2257](https://github.com/sgl-project/sglang-omni/pull/2257) | [title.txt](2257/title.txt) | [body.md](2257/body.md) | `main` |
| B | [#1499](https://github.com/sgl-project/sglang-omni/pull/1499) | [title.txt](1499/title.txt) | [body.md](1499/body.md) | #2257 |
| C | [#1500](https://github.com/sgl-project/sglang-omni/pull/1500) | [title.txt](1500/title.txt) | [body.md](1500/body.md) | #1499 |
| E | [#1502](https://github.com/sgl-project/sglang-omni/pull/1502) | [title.txt](1502/title.txt) | [body.md](1502/body.md) | #1500 |
| F | [#1486](https://github.com/sgl-project/sglang-omni/pull/1486) | [title.txt](1486/title.txt) | [body.md](1486/body.md) | #1502 |
| H | [#1501](https://github.com/sgl-project/sglang-omni/pull/1501) | [title.txt](1501/title.txt) | [body.md](1501/body.md) | #1502 |

The stack is **A → B → C → E → (F, H)**. E includes the shared relay lifecycle;
H includes generic stage-SP support. F and H are sibling optimization branches.
These are code dependencies, not a requirement to finish all optimizations
before merging basic generation support.

Roadmap [#2207](https://github.com/sgl-project/sglang-omni/issues/2207) has its own
[title.txt](roadmap-2207/title.txt) and [body.md](roadmap-2207/body.md). It continues
the work tracked in #445 and prioritizes core functionality before optimization.

## Archived proposals

The following files are retained unchanged as historical records, not active
publication drafts or prerequisites:

| Proposal | Superseded by | Archived title | Archived description |
| --- | --- | --- | --- |
| [#1487](https://github.com/sgl-project/sglang-omni/pull/1487), shared relay | E/#1502 | [title.txt](1487/title.txt) | [body.md](1487/body.md) |
| [#1490](https://github.com/sgl-project/sglang-omni/pull/1490), generic stage SP | H/#1501 | [title.txt](1490/title.txt) | [body.md](1490/body.md) |

This records their place in the contribution plan; it does not claim their
GitHub open/closed state has been changed.

## Editing and publishing

1. Edit the relevant `title.txt` and `body.md` here. Keep code changes on the
   implementation branches.
2. Review the wording against the actual code and immediate prerequisite.
   Validation placeholders must be replaced with exact tested commits,
   environments, results, and remaining limitations before publication.
3. Before publishing, read the upstream PR or issue again and reconcile any
   concurrent edits. Do not replace newer collaborator text without review.
4. The PR/issue author or another user with the necessary upstream permission
   publishes the approved text, then reads it back to verify it.

Title files contain only the title; body files contain only Markdown for the
PR or issue. Contributors are preserved in each PR description. The native
`data[]`, single-image chat `message.image`, and interleaved chat
`message.segments` with plain-text `message.content` contracts must not be conflated.

No publishing script, workflow, credentials, or additional GitHub permissions
are configured by this branch. Do not open an upstream code PR for it or merge
it into `main` or feature branches. Keep private development material out of
this directory.

## Initial snapshot

`snapshot.json` is the unchanged metadata from the initial seven-PR export.
Its timestamps, titles, branch names, and Draft states are historical, not
current status. It is not rewritten when these working copies are edited.

The original exports preserved line endings, blank lines, and final-newline
presence. The scoped `.gitattributes` prevents Git normalization but does not
prevent editors or formatting hooks from rewriting archived files. Keep the
#1487/#1490 title and body files byte-for-byte unchanged.
