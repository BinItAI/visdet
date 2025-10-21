# Contributing

We appreciate all contributions to improve MMDetection. Please follow the guidelines below.

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
- `pre-commit` hooks for automated checks

To set up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Testing

Before submitting a pull request, make sure all tests pass:

```bash
pytest tests/
```

## Documentation

If you add new features, please update the documentation accordingly.

## Questions?

If you have any questions, please open an issue or reach out to the maintainers.
