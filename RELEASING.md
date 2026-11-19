# Releasing

Releases are built by [`.github/workflows/release.yml`](.github/workflows/release.yml). Pushing a tag such as `v0.1.0` builds wheels for Linux (x86_64 and ARM), macOS (Apple silicon and Intel) and Windows, plus a source distribution. It installs the Linux wheel and runs the tests and evals against it. If they pass, it creates a GitHub release for the tag, attaches the wheels and the source distribution, and takes the notes from that version's section of `CHANGELOG.md`.

Publishing to PyPI and crates.io is off until the one-time setup below is done.

## Each release

1. Update `version` in `Cargo.toml` (workspace) and `pyproject.toml`.
2. Rename the top section of `CHANGELOG.md` to the new version and date.
3. Update the version in the install commands in `README.md`.
4. Commit, then tag and push:

   ```bash
   git tag v0.1.0
   ```

   ```bash
   git push origin v0.1.0
   ```

5. Watch the *Release* workflow. If any wheel, test or eval fails, nothing is released.

To build wheels without releasing, run the workflow by hand from the *Actions* tab.

## One-time setup for PyPI and crates.io

**PyPI, with trusted publishing (no token stored anywhere):**

1. Sign in at pypi.org and open *Your account → Publishing*.
2. Add a pending publisher: project `mnemosoma`, owner `shakehasan`, repository `mnemosoma`, workflow `release.yml`, environment `pypi`.
3. In the GitHub repository, create an environment named `pypi` (*Settings → Environments*).
4. Add a repository variable named `PUBLISH_TO_PYPI` with the value `true` (*Settings → Secrets and variables → Actions → Variables*).

**crates.io:**

1. Sign in at crates.io with GitHub and create an API token with the `publish-new` and `publish-update` scopes.
2. In the GitHub repository, create an environment named `crates-io` and add the token as a secret named `CARGO_REGISTRY_TOKEN`.
3. Add a repository variable named `PUBLISH_TO_CRATES_IO` with the value `true`.

The next tag pushed after that is published to the registries as well.
