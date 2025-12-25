#!/bin/bash
# Script to fix failing prek checks on all open PRs
# Usage: ./scripts/fix_failing_prs.sh [--dry-run]

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

WORKTREE_ROOT="/Users/georgepearse/visdet-worktrees"
MAIN_DIR="$WORKTREE_ROOT/main"

cd "$MAIN_DIR"

echo "Fetching open PRs with failing checks..."

# Get all open PRs
prs=$(gh pr list --state open --json number,headRefName,title --limit 100)

# Parse each PR
echo "$prs" | jq -c '.[]' | while read -r pr; do
    pr_number=$(echo "$pr" | jq -r '.number')
    branch=$(echo "$pr" | jq -r '.headRefName')
    title=$(echo "$pr" | jq -r '.title')

    echo ""
    echo "=========================================="
    echo "Checking PR #$pr_number: $title"
    echo "Branch: $branch"
    echo "=========================================="

    # Check if prek is failing
    prek_status=$(gh pr checks "$pr_number" 2>&1 | grep -E "^prek\s+" | awk '{print $2}' || echo "unknown")

    if [[ "$prek_status" == "pass" ]]; then
        echo "✅ prek is passing - skipping"
        continue
    elif [[ "$prek_status" == "pending" ]]; then
        echo "⏳ prek is pending - skipping"
        continue
    elif [[ "$prek_status" == "fail" ]]; then
        echo "❌ prek is failing - attempting fix..."
    else
        echo "❓ prek status unknown ($prek_status) - checking anyway..."
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would fix PR #$pr_number ($branch)"
        continue
    fi

    # Create or use existing worktree for this branch
    worktree_dir="$WORKTREE_ROOT/fix-pr-$pr_number"

    if [[ -d "$worktree_dir" ]]; then
        echo "Using existing worktree at $worktree_dir"
        cd "$worktree_dir"
        git fetch origin "$branch"
        git checkout "$branch"
        git pull origin "$branch"
    else
        echo "Creating worktree at $worktree_dir"
        cd "$MAIN_DIR"
        git fetch origin "$branch"
        git worktree add "$worktree_dir" "$branch" || {
            echo "Failed to create worktree, trying to checkout existing branch"
            git worktree add "$worktree_dir" -b "temp-fix-$pr_number" "origin/$branch"
        }
        cd "$worktree_dir"
    fi

    # Run ruff fix
    echo "Running ruff fix..."
    uv run ruff check --fix . 2>&1 || true

    # Check if there are changes
    if git diff --quiet && git diff --cached --quiet; then
        echo "No changes needed from ruff"
    else
        echo "Committing ruff fixes..."
        git add -A
        git commit -m "fix: auto-fix ruff lint errors

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>" || echo "Nothing to commit"

        echo "Pushing fixes..."
        git push origin "$branch"
        echo "✅ Pushed fixes for PR #$pr_number"
    fi

    # Clean up worktree
    cd "$MAIN_DIR"
    echo "Cleaning up worktree..."
    git worktree remove "$worktree_dir" --force 2>/dev/null || true

    echo "Done with PR #$pr_number"
done

echo ""
echo "=========================================="
echo "All PRs processed!"
echo "=========================================="
