# GitHub Actions (disabled)

This fork does **not** use GitHub Actions. All workflow files under `.github/workflows/` were removed so that **push and pull_request do not start any CI**.

Build wheels and run tests **locally**:

- `WindowsWhlBuilder_cuda.bat`
- `md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md`

Releases: create GitHub Releases manually and attach wheel artifacts.

Do not re-add `ci.yml`, `pre-commit.yaml`, or upstream self-hosted GPU workflows without explicit owner approval.
