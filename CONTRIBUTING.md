# Contributing to Mosaic

Thank you for your interest in contributing to Mosaic! This document provides guidelines and instructions for contributors.

## Getting Started

### 1. Fork and Clone

1. **Fork the repository** on GitHub by clicking the "Fork" button
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/tareksanger/mosaic.git
   cd mosaic-mind
   ```

3. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/mosaic.git
   ```

### 2. Development Setup

This project uses [pre-commit](https://pre-commit.com/) to ensure code quality checks are automatically run before each commit.

#### One-time setup

Install the development dependencies and activate pre-commit hooks:

```bash
uv pip install --dev
pre-commit install
```

After this setup, all future commits will automatically run formatting and lint checks.

## Making Changes

### 1. Create a Feature Branch

Always create a new branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests

### 2. Make Your Changes

Write your code following the project's coding standards and guidelines.

### 3. Commit Your Changes

Use conventional commit messages:

```bash
git add .
git commit -m "feat: add new agent orchestration feature"
git commit -m "fix: resolve LLM timeout issue"
git commit -m "docs: update API documentation"
```

**Commit message format:**
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

## Submitting Changes

### 1. Open a Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request" for your branch
3. Fill out the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if applicable)

### 2. Code Review

- All PRs require review before merging
- Address any feedback from reviewers
- Keep PRs focused and reasonably sized
- Update documentation for new features

## Code Quality

### Pre-commit Hooks

The following checks run automatically on each commit:
- **Ruff**: Code formatting and linting
- **MyPy**: Type checking
- **Pyright**: Additional type checking for IDE compatibility

### Manual Checks

You can run quality checks manually:

```bash
# Format and lint code
ruff check --fix
ruff format

# Type checking
mypy src/
pyright

# Run tests
pytest
```

## Project Structure

```
src/mosaic/
├── core/
│   ├── ai/
│   │   ├── embedding/     # Embedding models
│   │   └── llm/          # LLM implementations
│   └── common/           # Shared utilities
└── __init__.py
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=src/mosaic
```

### Writing Tests

- Write tests for new features
- Ensure existing tests pass
- Follow the existing test patterns
- Use descriptive test names

## Type Checking

The project uses strict type checking. Ensure all code is properly typed:

```bash
mypy src/
pyright
```

## Dependencies

- **Runtime**: Python 3.9+
- **Development**: See `pyproject.toml` for dev dependencies
- **Package Manager**: `uv` (recommended) or `pip`

## Getting Help

- Check existing issues and PRs
- Join discussions in issues
- Ask questions in issues with the "question" label

## Code of Conduct

Please be respectful and inclusive in all interactions. We aim to create a welcoming environment for all contributors.

Thank you for contributing to Mosaic Mind! 