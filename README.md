

## Development Setup

This project uses [pre-commit](https://pre-commit.com/) to ensure code quality checks are automatically run before each commit.

### One-time setup

Install the development dependencies and activate pre-commit hooks:

```bash
uv pip install --dev
pre-commit install
```

After this setup, all future commits will automatically run formatting and lint checks.

## Releasing a New Version

To release a new version to PyPI:

1. Bump the version number in `pyproject.toml`:
   ```toml
   version = "0.x.y"
   ```

2. Commit the change:
   ```bash
   git commit -am "Bump version to 0.x.y"
   ```

3. Tag the release and push:
   ```bash
   git tag v0.x.y
   git push origin v0.x.y
   ```

This will trigger the GitHub Actions workflow to build and publish the package to PyPI using `uv publish`.