#!/usr/bin/env bash
# Daily fork sync: pull andrewyng/openworker (upstream) into our fork's `main`.
#
# Runs from cron, so it never changes which branch you are on, and it touches the
# working tree in exactly one case: when `main` itself is checked out, where it runs
# `git merge --ff-only` (files change under an open session, but only ever forward,
# and a dirty tree with conflicting edits aborts the merge rather than losing work).
# On any other branch, `git fetch <remote> main:main` moves the ref without going near
# your files. Both forms REFUSE a non-fast-forward, which is the point: divergence is a
# human decision, not something a cron job should merge at 03:40.
#
# Remotes are resolved by URL, not by name — in this clone `origin` is upstream and
# `fork` is ours, which is the opposite of the usual convention.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${OPENWORKER_SYNC_LOG:-$HOME/.local/state/openworker-sync.log}"
UPSTREAM_URL_MATCH="andrewyng/openworker"
FORK_URL_MATCH="clintoncodewell/openworker"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) sync-upstream ==="

cd "$REPO"

remote_matching() {
  # First remote whose fetch URL contains $1. Empty output = not found.
  git remote -v | awk -v pat="$1" '$3 == "(fetch)" && index($2, pat) { print $1; exit }'
}

UPSTREAM="$(remote_matching "$UPSTREAM_URL_MATCH")"
FORK="$(remote_matching "$FORK_URL_MATCH")"

if [ -z "$UPSTREAM" ]; then
  echo "FAIL: no remote points at $UPSTREAM_URL_MATCH"
  exit 1
fi

git fetch --quiet "$UPSTREAM"
BEFORE="$(git rev-parse main 2>/dev/null || echo none)"
AFTER="$(git rev-parse "$UPSTREAM/main")"

if [ "$BEFORE" = "$AFTER" ]; then
  echo "up to date at ${AFTER:0:9}"
elif [ "$(git rev-parse --abbrev-ref HEAD)" = "main" ]; then
  # main is checked out, so the ref is pinned by the worktree — merge instead. Tested
  # with `if !` rather than a bare call: under `set -e` a failing merge would exit here,
  # so the diverged message below would never print and the peer sync would be skipped
  # for a reason that has nothing to do with the peer.
  if git merge --ff-only "$UPSTREAM/main"; then
    echo "fast-forwarded checked-out main ${BEFORE:0:9} -> ${AFTER:0:9}"
  else
    echo "FAIL: main is checked out and would not fast-forward — commit, stash or resolve by hand"
    SYNC_FAILED=1
  fi
elif git fetch "$UPSTREAM" main:main 2>&1; then
  echo "fast-forwarded main ${BEFORE:0:9} -> ${AFTER:0:9}"
else
  echo "FAIL: main has diverged from $UPSTREAM/main — resolve by hand"
  SYNC_FAILED=1
fi

# Only the primary machine pushes. A peer clone is a build mirror that needs to PULL;
# it usually has no GitHub credential, and its push would be a no-op anyway since the
# primary already pushed the same commit.
if [ "${OPENWORKER_SYNC_IS_PEER:-}" = "1" ]; then
  echo "peer run — pull only, the primary owns the push"
elif [ -n "$FORK" ]; then
  git push --quiet "$FORK" main:main
  echo "pushed main to $FORK"
else
  echo "note: no fork remote found ($FORK_URL_MATCH); local main only"
fi

BEHIND="$(git rev-list --count "HEAD..$UPSTREAM/main")"
echo "current branch $(git rev-parse --abbrev-ref HEAD) is $BEHIND commit(s) behind upstream main"

# The Mac holds a second clone (~/dev/openworker) that the desktop app is built from, so
# it needs the same sync or the app drifts behind upstream. Run this same script there
# rather than pushing state across: each clone stays its own git repo, and the ff-only
# rules above protect the Mac's tree exactly as they protect this one.
#
# Guarded by a short reachability probe: a sleeping laptop is the normal case at 03:40 and
# must not fail the run. Set OPENWORKER_SYNC_PEER= to skip it entirely.
# Runs even if this machine's own merge failed: the peer is a separate clone and its
# sync is not conditional on ours.
PEER="${OPENWORKER_SYNC_PEER-mac}"
PEER_PATH="${OPENWORKER_SYNC_PEER_PATH:-~/dev/openworker}"
if [ -n "$PEER" ] && [ "${OPENWORKER_SYNC_IS_PEER:-}" != "1" ]; then
  # `--` before the destination: a peer name starting with "-" would otherwise be read by
  # ssh as an option rather than a host.
  if ssh -o ConnectTimeout=8 -o BatchMode=yes -- "$PEER" true 2>/dev/null; then
    echo "--- $PEER ---"
    # OPENWORKER_SYNC_IS_PEER stops the peer from trying to sync back here forever.
    ssh -- "$PEER" "cd $PEER_PATH && OPENWORKER_SYNC_IS_PEER=1 OPENWORKER_SYNC_LOG=/dev/stdout ./scripts/sync-upstream.sh" \
      || echo "note: $PEER sync failed (see lines above) — this machine is still synced"
  else
    echo "note: $PEER unreachable (asleep?) — skipped"
  fi
fi

exit "${SYNC_FAILED:-0}"
