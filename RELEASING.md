# Releasing

## Every release

1. On a branch, bump the version in `pyproject.toml`, `contilab/__init__.py` and `CITATION.cff`
   (`version`, `date-released`). Update the BibTeX `version` in `README.md` and `docs/wiki/Citing.md`.
2. In `CHANGELOG.md`, rename `## Unreleased` to `## X.Y.Z — YYYY-MM-DD`.
3. Open a PR, wait for CI, merge.
4. Push the tag `vX.Y.Z` on the merge commit, or run **Actions → Release → Run workflow** with tag `vX.Y.Z` on
   `main`. The [release workflow](.github/workflows/release.yml) then:
   - checks the tag against the `pyproject.toml` version
   - writes the notes from that version's CHANGELOG section
   - builds the sdist and wheel and attaches them to a GitHub release
   - uploads them to PyPI, once that is enabled (below)

## One-time setup

### PyPI (trusted publishing, no stored token)

1. On <https://pypi.org/manage/account/publishing/>, add a pending publisher:
   - project `contilab`
   - owner `JonPark0`
   - repository `contilab`
   - workflow `release.yml`
   - environment `pypi`
2. In the repository settings:
   - create the environment `pypi` (Settings → Environments)
   - add the repository variable `PUBLISH_PYPI` = `true` (Settings → Secrets and variables → Actions →
     Variables)

The next release is then uploaded to PyPI as well. Until the variable is set, the PyPI job is skipped.

### Zenodo (DOI)

1. Sign in to <https://zenodo.org> with GitHub, open **GitHub** in the account menu, and switch on
   `JonPark0/contilab`.
2. Publish the next release. Zenodo archives it and mints a DOI, taking the metadata from `CITATION.cff`.
3. Add the concept DOI (the one that covers all versions) to `CITATION.cff` (`doi:`) and to the BibTeX entries,
   and add the DOI badge to the README.
