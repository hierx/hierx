# Contributing to hierx

Thank you for your interest in contributing to `hierx`!

## Bug Reports

Please open an issue on GitHub with:
- A minimal reproducible example
- Expected vs actual behaviour
- Python version and OS

## Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Install in development mode: `pip install -e ".[dev]"`
4. Make your changes
5. Run tests: `pytest tests/`
6. Run linter: `ruff check .`
7. Open a PR against `main`

## Development Setup

```bash
git clone https://github.com/hierx/hierx.git
cd hierx
pip install -e ".[dev,plot]"
```

## Code Style

- Format with `ruff format .`
- Lint with `ruff check .`
- Follow existing patterns in the codebase

## Testing

- All new features should include tests in `tests/`
- Run the full suite with `pytest tests/ -v`
- Slow tests are marked with `@pytest.mark.slow`
