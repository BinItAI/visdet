# Contributing

We appreciate all contributions to improve VisDet. Please follow the guidelines below.

## Workflow

1. Fork and clone the repository
2. Create a new branch for your changes
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## Code Style

We use the following tools to maintain code quality:

- `ruff` for linting and formatting
- `pyright` for type checking
- `prek` hooks for automated checks (faster Rust-based alternative to pre-commit)

To set up the development environment:

```bash
# Clone the repository
git clone <your-repository-url>
cd visdet

# Install all dependencies (including dev dependencies)
uv sync

# Set up prek hooks
uv run prek install
```

## Testing

Before submitting a pull request, make sure all tests pass:

```bash
uv run pytest tests/
```

## Documentation

If you add new features, please update the documentation accordingly.

## Questions?

If you have any questions, please open an issue or reach out to the maintainers.
