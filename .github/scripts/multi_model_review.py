#!/usr/bin/env python3
"""Multi-model consensus code review for PRs."""

import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel


@dataclass
class ModelConfig:
    name: str
    model_id: str


# Production models via OpenRouter
MODELS = [
    ModelConfig("Claude Sonnet 4", "anthropic/claude-sonnet-4"),
    ModelConfig("GPT-4o", "openai/gpt-4o"),
    ModelConfig("Gemini 2.0 Flash", "google/gemini-2.0-flash-001"),
]

REVIEW_PROMPT = """You are reviewing Python code for the visdet object detection library.

Review for:
1. Security vulnerabilities (SQL injection, command injection, unsafe deserialization)
2. Type safety issues (missing annotations, incorrect types)
3. Error handling problems (bare except, missing error handling)
4. Performance issues (inefficient loops, memory leaks)
5. Code quality (unused imports, dead code, unclear logic)

For each issue found, respond with a line starting with "ISSUE:" followed by a brief description.
If no issues, respond with "NO ISSUES FOUND".
"""


class ReviewResult(BaseModel):
    file: str
    models_passed: int
    total_models: int
    consensus_issues: list[str]
    unanimous_issues: list[str]
    all_issues: list[str]


async def query_model(client: AsyncOpenAI, model: ModelConfig, code: str) -> list[str]:
    """Query a single model and extract issues."""
    try:
        response = await client.chat.completions.create(
            model=model.model_id,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": REVIEW_PROMPT},
                {"role": "user", "content": f"Review this code:\n\n```python\n{code}\n```"},
            ],
        )
        text = response.choices[0].message.content or ""

        # Extract issues
        issues = []
        for line in text.split("\n"):
            if line.strip().upper().startswith("ISSUE:"):
                issues.append(line.split(":", 1)[1].strip())
        return issues
    except Exception as e:
        print(f"Error querying {model.name}: {e}")
        return []


async def review_file(client: AsyncOpenAI, file_path: str, code: str) -> ReviewResult:
    """Review a file with all models and compute consensus."""
    tasks = [query_model(client, model, code) for model in MODELS]
    all_results = await asyncio.gather(*tasks)

    # Aggregate issues
    all_issues_flat = [issue for issues in all_results for issue in issues]
    issue_counts = Counter(all_issues_flat)

    total_models = len(MODELS)
    majority = total_models // 2 + 1

    consensus = [issue for issue, count in issue_counts.items() if count >= majority]
    unanimous = [issue for issue, count in issue_counts.items() if count == total_models]

    models_with_issues = sum(1 for issues in all_results if len(issues) > 0)

    return ReviewResult(
        file=file_path,
        models_passed=total_models - models_with_issues,
        total_models=total_models,
        consensus_issues=consensus,
        unanimous_issues=unanimous,
        all_issues=list(issue_counts.keys()),
    )


async def main():
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    changed_files = Path("/tmp/changed_files.txt").read_text().strip().split("\n")
    changed_files = [f for f in changed_files if f.strip()]

    results = {}
    for file_path in changed_files[:10]:  # Limit to 10 files
        try:
            code = Path(file_path).read_text()
            if len(code) > 15000:
                code = code[:15000]
            result = await review_file(client, file_path, code)
            results[file_path] = result.model_dump()
            print(f"Reviewed {file_path}: {len(result.consensus_issues)} consensus issues")
        except Exception as e:
            print(f"Error reviewing {file_path}: {e}")

    Path("/tmp/review_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
