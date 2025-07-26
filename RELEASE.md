# Release Guide

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

## Package Information

- **Package Name**: `mosaic-mind`
- **PyPI Project**: https://pypi.org/project/mosaic-mind/
- **Install Command**: `pip install mosaic-mind`

## Version Management

The project uses semantic versioning:
- `0.x.y-alpha.z` for alpha releases
- `0.x.y` for stable releases
- `x.y.z` for major releases

## Automated Publishing

The GitHub Actions workflow automatically:
- Builds the package using `hatchling`
- Runs tests and linting
- Publishes to PyPI when a tag is pushed
