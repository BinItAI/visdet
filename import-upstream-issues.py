#!/usr/bin/env python3
"""
Import open issues from the mmdetection repository.

This script fetches all open issues from the upstream mmdetection repository and creates
corresponding issues in the visdet repository, with links back to the originals.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm  # type: ignore[import-untyped]

UPSTREAM_REPO = "open-mmlab/mmdetection"
FORK_REPO = "BinItAI/visdet"
UPSTREAM_LABEL = "from-mmdetection"
TRACKING_FILE = "imported-issues.json"
RATE_LIMIT_DELAY = 1.5  # seconds between issue creation


def run_gh_command(args: list[str]) -> str:
    """Run a gh CLI command and return the output."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running gh command: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise


def ensure_label_exists() -> None:
    """Ensure the 'from-mmdetection' label exists in the repository."""
    print(f"Ensuring '{UPSTREAM_LABEL}' label exists...")
    try:
        # Try to create the label
        run_gh_command(
            [
                "label",
                "create",
                UPSTREAM_LABEL,
                "--repo",
                FORK_REPO,
                "--description",
                "Issue imported from upstream open-mmlab/mmdetection repository",
                "--color",
                "0366d6",
            ]
        )
        print(f"✓ Created '{UPSTREAM_LABEL}' label")
    except subprocess.CalledProcessError:
        # Label might already exist
        print(f"✓ '{UPSTREAM_LABEL}' label already exists")


def fetch_upstream_issues() -> list[dict[str, Any]]:
    """Fetch all open issues from the upstream repository."""
    print(f"Fetching open issues from {UPSTREAM_REPO}...")

    all_issues = []
    page_size = 1000
    page = 1

    while True:
        print(f"  Fetching page {page} (limit: {page_size})...")
        try:
            # Calculate the starting issue number based on page
            # Note: We need to use --limit with a large enough value to get all issues
            # GitHub CLI doesn't have great pagination support, so we'll try 10000
            output = run_gh_command(
                [
                    "issue",
                    "list",
                    "--repo",
                    UPSTREAM_REPO,
                    "--state",
                    "open",
                    "--limit",
                    "10000",  # Increased limit to fetch all issues
                    "--json",
                    "number,title,body,labels,createdAt,updatedAt,url",
                ]
            )
            issues = json.loads(output)
            all_issues = issues  # Since we're fetching all at once now
            break
        except subprocess.CalledProcessError as e:
            print(f"  Error fetching issues: {e}")
            raise

    print(f"✓ Found {len(all_issues)} open issues")
    return all_issues


def load_tracking_data() -> dict[str, Any]:
    """Load the tracking file to see which issues have already been imported."""
    tracking_path = Path(TRACKING_FILE)
    if tracking_path.exists():
        with open(tracking_path) as f:
            return json.load(f)
    return {"imported": {}}


def save_tracking_data(data: dict[str, Any]) -> None:
    """Save the tracking file."""
    with open(TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=2)


def format_issue_body(issue: dict[str, Any]) -> str:
    """Format the issue body with a link to the original."""
    original_url = issue["url"]
    created_at = datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00"))
    updated_at = datetime.fromisoformat(issue["updatedAt"].replace("Z", "+00:00"))

    body_parts = [
        f"**Original issue:** {original_url}",
        f"**Created:** {created_at.strftime('%Y-%m-%d')}",
        f"**Last updated:** {updated_at.strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        issue.get("body", "").strip() or "*No description provided.*",
    ]

    return "\n".join(body_parts)


def create_issue(issue: dict[str, Any]) -> str:
    """Create an issue in the fork repository."""
    upstream_number = issue["number"]
    title = issue["title"]
    body = format_issue_body(issue)

    print(f"  Creating issue #{upstream_number}: {title[:60]}...")

    # Create the issue without labels first
    cmd = [
        "issue",
        "create",
        "--repo",
        FORK_REPO,
        "--title",
        title,
        "--body",
        body,
        "--label",
        UPSTREAM_LABEL,  # Always add the upstream label
    ]

    output = run_gh_command(cmd)
    fork_issue_url = output.strip()

    # Try to add other labels if they exist
    # Extract issue number from URL
    issue_number = fork_issue_url.split("/")[-1]

    if issue.get("labels"):
        for label in issue["labels"]:
            label_name = label["name"]
            try:
                # Try to add the label
                run_gh_command(
                    [
                        "issue",
                        "edit",
                        issue_number,
                        "--repo",
                        FORK_REPO,
                        "--add-label",
                        label_name,
                    ]
                )
            except subprocess.CalledProcessError:
                # Label doesn't exist in fork, skip it
                pass

    print(f"  ✓ Created: {fork_issue_url}")
    return fork_issue_url


def main() -> None:
    """Main entry point."""
    print("=" * 70)
    print("Import Upstream Issues")
    print("=" * 70)
    print()

    # Ensure label exists
    ensure_label_exists()
    print()

    # Fetch upstream issues
    upstream_issues = fetch_upstream_issues()
    print()

    # Load tracking data
    tracking_data = load_tracking_data()
    imported = tracking_data.get("imported", {})

    # Filter out already imported issues
    to_import = [issue for issue in upstream_issues if str(issue["number"]) not in imported]

    if not to_import:
        print("✓ All issues have already been imported!")
        return

    print(f"Found {len(to_import)} issues to import")
    print(f"Already imported: {len(imported)}")
    print()

    # Import issues
    print("Starting import...")
    print()

    success_count = 0
    failed = []

    for issue in tqdm(to_import, desc="Importing issues", unit="issue"):
        upstream_number = issue["number"]
        try:
            fork_url = create_issue(issue)

            # Track the import
            imported[str(upstream_number)] = {
                "fork_url": fork_url,
                "imported_at": datetime.now().isoformat(),
                "title": issue["title"],
            }

            # Save after each successful import
            tracking_data["imported"] = imported
            save_tracking_data(tracking_data)

            success_count += 1

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            tqdm.write(f"  ✗ Failed to create issue #{upstream_number}: {e}")
            failed.append((upstream_number, str(e)))
            # Continue with next issue

    print()
    print("=" * 70)
    print("Import Complete")
    print("=" * 70)
    print(f"Successfully imported: {success_count}/{len(to_import)}")
    print(f"Total imported (all time): {len(imported)}")

    if failed:
        print(f"\nFailed issues ({len(failed)}):")
        for number, error in failed:
            print(f"  #{number}: {error}")

    print()
    print(f"Tracking data saved to: {TRACKING_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)
