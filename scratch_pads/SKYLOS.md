# Skylos Integration Guide

This project uses **Skylos**, a fast and comprehensive static analysis tool for detecting dead code and security vulnerabilities in Python projects.

## What is Skylos?

Skylos is a modern Python linter that:
- **Detects dead code**: Unused imports, classes, and functions
- **Scans for security vulnerabilities**: SQL injection, command injection, eval/exec, unsafe pickle/YAML, weak hashing, and path traversal
- **Framework-aware**: Understands Flask, Django, and FastAPI route detection
- **Fast**: Analyzes large codebases efficiently
- **Safe removal**: Uses LibCST for syntactically correct code modifications

**Repository**: https://github.com/duriantaco/skylos

## Installation

Skylos is included in the development dependencies. Install it with:

```bash
uv sync
```

## Usage

### Basic Scan

Run a comprehensive scan on the entire project:

```bash
skylos .
```

### Table Format (Human-Readable)

```bash
skylos . --table
```

### JSON Format (Machine-Readable)

```bash
skylos . --json
```

### Security-Only Scan

Focus on security vulnerabilities:

```bash
skylos . --danger
```

### Secrets Detection

Scan for exposed API keys and credentials:

```bash
skylos . --secrets
```

### Exclude Directories

```bash
skylos . --exclude=tests,archive,configs
```

### Interactive Mode

Selectively remove dead code items:

```bash
skylos . --interactive
```

### Dry Run

Preview changes without applying them:

```bash
skylos . --select matchers --dry-run
```

## Configuration

Skylos is configured via `.skylosrc` file. Key settings:

- **exclude**: Directories to skip during scanning
- **security**: Enable/disable specific security checks
- **frameworks**: Framework-aware scanning (Flask, Django, FastAPI)
- **deadcode**: Dead code detection settings
- **output**: Output format preferences
- **confidence**: Confidence thresholds for findings

### Example Configuration

```yaml
exclude:
  - tests
  - archive
  - configs

security:
  enabled: true
  check_sql_injection: true
  check_command_injection: true
  check_eval_exec: true
  check_pickle_yaml: true

confidence:
  min_deadcode: 0.8
  min_security: 0.9
```

## prek Hook Integration

Skylos is automatically run as a prek hook. It will:
1. Scan for dead code and security issues
2. Report findings in table format
3. Exclude test files and configuration directories

To run prek manually:

```bash
prek run skylos --all-files
```

## CI/CD Integration

### GitHub Actions

Skylos runs automatically on:
- **Pull requests** to `master` or `main`
- **Pushes** to `master`, `main`, or feature branches

The workflow:
1. Runs security and dead code scans
2. Generates a detailed JSON report
3. Comments on PRs with summary findings
4. Uploads report as artifact for review

View results in:
- **PR comments**: Summary of findings
- **Artifacts**: Full JSON report in GitHub Actions

## Handling Skylos Findings

### Dead Code

False positives are common for:
- Framework magic (Flask routes, Django decorators)
- Dynamic imports
- Plugin systems

Exclude specific items:

```bash
skylos . --exclude-dead-code=matchers,util
```

### Security Issues

Review each security finding:

1. **Path Traversal (SKY-D215)**: Usually false positive for config file handling
2. **eval/exec (SKY-D201/D202)**: Can be legitimate for configuration systems
3. **Command Injection (SKY-D212)**: May be needed for CLI tools

Mark as reviewed but don't remove critical functionality.

## Benchmarks

Skylos performance compared to alternatives:

| Tool | Dead Code F1 | Security F1 |
|------|-------------|-------------|
| **Skylos** | 0.698 | High |
| Vulture | 0.367 | N/A |
| Flake8 | 0.244 | Limited |

## Resources

- **Official Docs**: https://github.com/duriantaco/skylos
- **Interactive Tool**: http://localhost:5090 (when running `skylos --interactive`)
- **Issue Reports**: https://github.com/duriantaco/skylos/issues

## Troubleshooting

### Skylos hangs on large files

Increase timeout or exclude problematic files:

```bash
skylos . --exclude=path/to/large/file.py
```

### Too many false positives

Adjust confidence thresholds in `.skylosrc`:

```yaml
confidence:
  min_deadcode: 0.9  # Higher = fewer false positives
  min_security: 0.95
```

### Pre-commit hook fails locally but passes in CI

Ensure you're using the same Python version:

```bash
python --version  # Should be 3.12+
uv sync           # Reinstall dependencies
```

## Integration Notes

This integration was added to:
- ✅ Catch dead code early in development
- ✅ Detect security vulnerabilities before production
- ✅ Improve code quality and maintainability
- ✅ Automate security scanning in CI/CD

The tool is configured conservatively to minimize false positives while catching real issues.
