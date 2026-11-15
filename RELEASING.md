# Releasing

Releases are built and published by [`.github/workflows/release.yml`](.github/workflows/release.yml). Pushing a tag such as `v0.1.0` builds wheels for Linux (x86_64 and ARM), macOS (Apple silicon and Intel) and Windows, plus a source distribution. It installs the Linux wheel, runs the tests and evals against it, and then publishes.

## One-time setup

**PyPI, with trusted publishing (no token stored anywhere):**

1. Sign in at pypi.org and open *Your account → Publishing*.
2. Add a pending publisher: project `mnemosoma`, owner `shakehasan`, repository `mnemosoma`, workflow `release.yml`, environment `pypi`.
3. In the GitHub repository, create an environment named `pypi` (*Settings → Environments*).

**crates.io:**

1. Sign in at crates.io with GitHub and create an API token with the `publish-new` and `publish-update` scopes.
2. In the GitHub repository, create an environment named `crates-io` and add the token as a secret named `CARGO_REGISTRY_TOKEN`.

## Each release

1. Update `version` in `Cargo.toml` (workspace) and `pyproject.toml`.
2. Commit, then tag and push:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. Watch the *Release* workflow. If any wheel, test or eval fails, nothing is published.

To build wheels without publishing, run the workflow by hand from the *Actions* tab.
