#!/usr/bin/env python3
"""
Import only open issues from the mmdetection repository.

This script fetches only open issues from the upstream mmdetection repository and creates
corresponding issues in the visdet repository, with links back to the originals.
Closed issues are not imported.
"""

import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm  # type: ignore[import-untyped]

UPSTREAM_REPO = "open-mmlab/mmdetection"
FORK_REPO = "BinItAI/visdet"
UPSTREAM_LABEL = "from-mmdetection"
TRACKING_FILE = "imported-issues.json"
RATE_LIMIT_DELAY = 10  # seconds between issue creation (increased to avoid GitHub secondary rate limits)


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
    """Fetch only open issues from the upstream repository using GraphQL pagination."""
    print(f"Fetching open issues from {UPSTREAM_REPO}...")

    owner, repo = UPSTREAM_REPO.split("/")
    all_issues = []
    has_next_page = True
    cursor = None
    page = 1

    while has_next_page:
        print(f"  Fetching page {page}...")

        # Build GraphQL query with pagination - filtering for OPEN issues only
        after_clause = f', after: "{cursor}"' if cursor else ""
        query = f"""
        {{
          repository(owner: "{owner}", name: "{repo}") {{
            issues(first: 100{after_clause}, states: OPEN, orderBy: {{field: CREATED_AT, direction: ASC}}) {{
              pageInfo {{
                hasNextPage
                endCursor
              }}
              nodes {{
                number
                title
                body
                url
                createdAt
                updatedAt
                state
                labels(first: 10) {{
                  nodes {{
                    name
                  }}
                }}
              }}
            }}
          }}
        }}
        """

        try:
            output = run_gh_command(["api", "graphql", "-f", f"query={query}"])
            result = json.loads(output)

            issues_data = result["data"]["repository"]["issues"]
            nodes = issues_data["nodes"]

            # Convert GraphQL format to match the CLI format
            for node in nodes:
                issue = {
                    "number": node["number"],
                    "title": node["title"],
                    "body": node.get("body", ""),
                    "url": node["url"],
                    "createdAt": node["createdAt"],
                    "updatedAt": node["updatedAt"],
                    "state": node["state"],
                    "labels": [{"name": label["name"]} for label in node["labels"]["nodes"]],
                }
                all_issues.append(issue)

            # Update pagination info
            page_info = issues_data["pageInfo"]
            has_next_page = page_info["hasNextPage"]
            cursor = page_info["endCursor"]
            page += 1

            print(f"    Fetched {len(nodes)} issues (total so far: {len(all_issues)})")

        except subprocess.CalledProcessError as e:
            print(f"  Error fetching issues: {e}")
            raise

    print(f"✓ Found {len(all_issues)} issues")
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
    state = issue.get("state", "OPEN")

    body_parts = [
        f"**Original issue:** {original_url}",
        f"**State in upstream:** {state}",
        f"**Created:** {created_at.strftime('%Y-%m-%d')}",
        f"**Last updated:** {updated_at.strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        issue.get("body", "").strip() or "*No description provided.*",
    ]

    return "\n".join(body_parts)


def check_issue_exists_in_fork(upstream_number: int) -> bool:
    """Check if an issue from this upstream number already exists in the fork."""
    try:
        # Search for issues in the fork that mention this upstream issue number
        search_query = f"repo:{FORK_REPO} is:issue {UPSTREAM_REPO}/issues/{upstream_number}"
        output = run_gh_command(["search", "issues", search_query, "--json", "number,title"])
        results = json.loads(output)
        return len(results) > 0
    except Exception:
        # If search fails, assume it doesn't exist to avoid blocking import
        return False


def create_issue(issue: dict[str, Any]) -> str:
    """Create an issue in the fork repository."""
    upstream_number = issue["number"]
    title = issue["title"]
    body = format_issue_body(issue)

    print(f"  Creating issue #{upstream_number}: {title[:60]}...")

    # Write body to a temporary file to avoid command line length limits
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(body)
        body_file = f.name

    try:
        # Create the issue using the body file
        cmd = [
            "issue",
            "create",
            "--repo",
            FORK_REPO,
            "--title",
            title,
            "--body-file",
            body_file,
            "--label",
            UPSTREAM_LABEL,  # Always add the upstream label
        ]

        run_gh_command(cmd)

        # gh issue create doesn't always output URL in non-interactive mode
        # Fetch the latest issue to get the number
        latest_output = run_gh_command(
            [
                "issue",
                "list",
                "--repo",
                FORK_REPO,
                "--limit",
                "1",
                "--state",
                "all",
                "--json",
                "number,url",
            ]
        )
        latest_issues = json.loads(latest_output)
        if not latest_issues:
            raise ValueError("No issues found after creation")

        issue_number = str(latest_issues[0]["number"])
        fork_issue_url = latest_issues[0]["url"]

    finally:
        # Clean up the temporary file
        Path(body_file).unlink(missing_ok=True)

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

    print(f"Found {len(to_import)} new issues to import")
    print(f"Already imported (tracked): {len(imported)}")
    print(f"Total open issues in upstream: {len(upstream_issues)}")
    print(f"Skipped (already imported): {len(upstream_issues) - len(to_import)}")
    print()

    # Import issues
    print("Starting import...")
    print()

    success_count = 0
    failed = []

    for issue in tqdm(to_import, desc="Importing issues", unit="issue"):
        upstream_number = issue["number"]
        try:
            # Double-check if issue already exists in fork (safety check)
            if check_issue_exists_in_fork(upstream_number):
                tqdm.write(f"  ⚠ Issue #{upstream_number} already exists in fork, skipping...")
                # Add to tracking file even if it wasn't there before
                imported[str(upstream_number)] = {
                    "fork_url": "unknown (existed before tracking)",
                    "imported_at": datetime.now().isoformat(),
                    "title": issue["title"],
                }
                tracking_data["imported"] = imported
                save_tracking_data(tracking_data)
                continue

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
