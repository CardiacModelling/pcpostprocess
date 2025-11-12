# Contributing to pcpostprocess

Contributions to pcpostprocess are always welcome!

We use [GIT](https://en.wikipedia.org/wiki/Git) and [GitHub](https://en.wikipedia.org/wiki/GitHub) to coordinate our work.
When making changes, we try to follow the procedure below.

1. **Discuss proposed changes before starting any work.**
   Before coding, always create an [issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/learning-about-issues/about-issues) and disucss the proposed work.
   Something similar may already exist, be under development, or have been proposed and rejected - so this can save valuable time.

2. **Always work on branches**.
   Once the idea has been agreed upon, start coding in a new _branch_.
   If you're in the CardiacModelling team, you can create a [branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository) directly in the main pcpostprocess repository.
   For outside contributions, you'll first need to [create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

   There are no rules for branch names, but try to make them relate to the issue, e.g. by including the issue number.

   Commit your changes to your branch with useful, descriptive commit messages that will still make sense in years to come.

3. **Conform to style guidelines, and document every class, method, and argument.**
   For more information, see below.
   
   **Note: as of 2025-11-11, we are still in the process of making `pcpostprocess` confirm to this rule.**
   
4. **Test locally, and ensure 100% test coverage**
    For more information, see below.

    **Note: as of 2025-11-11, we are stillin the process of making `pcpostprocess` confirm to this rule.**

4. **Discuss code in a PR**
   When your code is finished, or warrants discussion, create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) (PR).
   In your branch, update the [Changelog](./CHANGELOG.md) with a link to this PR and a concise summary of the changes.
   Finally, request a review of the code.

## Project structure

`pcpostprocess` is written in [Python 3](https://en.wikipedia.org/wiki/Python_(programming_language)), but at the moment has a few **non-python dependencies** i.e. latex.

## Developer installation

**TODO: Once there is a "user" way to install, move the git clone etc. information here [#105](https://github.com/CardiacModelling/pcpostprocess/issues/105).**

```
pip install -e .[test]
```

## Style guidelines

Style checking is done with [flake8](http://flake8.pycqa.org/en/latest/).

To run locally, use
```
$ flake8
```

Until Flake8 configuration supports [pyproject.toml](https://github.com/PyCQA/flake8/issues/234), it will be configured through [.flake8](./.flake8) ([syntax](http://flake8.pycqa.org/en/latest/user/configuration.html)).

In addition to the rules checked by flake8, we try to use single quotes (`'`) for strings, rather than double quotes (`"`) (but `"""` for docstrings).

### Spelling

Class, method, and argument names are in UK english.

### Import ordering

Import ordering is tested with [isort](https://pycqa.github.io/isort/index.html).

To run locally, use
```
isort --check-only --verbose ./pcpostprocess ./tests/
```

Isort is configured in [pyproject.toml](./pyproject.toml) under the section `tool.isort`.

## Documentation

Every method and every class should have a [docstring](https://www.python.org/dev/peps/pep-0257/) that describes in plain terms what it does, and what the expected input and output is.
The only exception are unit test methods starting with `test_` - unit test classes and other methods in unit tests should all have docstrings.

Each docstring should start with a one-line explanation.
If more explanation is needed, this one-liner is followed by a blank line and more information in the following paragraphs.

**TODO: READTHEDOCS [#60](https://github.com/CardiacModelling/pcpostprocess/issues/60)**

**TODO: SYNTAX, RUNNING LOCALLY, ETC**

**TODO: EXAMPLES**

```
cd doc
make clean
make html
```

[reStructuredText](http://docutils.sourceforge.net/docs/user/rst/quickref.html)
[Sphinx](http://www.sphinx-doc.org/en/stable/)

UK english

## Testing

To run all unit tests, locally:

```
$ python3 -m unittest
```

To run a single test, use
```
$ python3 -m unittest TestClass.test_method
```

### Unit tests

Testing is done with [unittest](https://docs.python.org/3/library/unittest.html).

Each method in `pcpostprocess` should have a unit test.

Tests should always aim to compare generated output with reference values, instead of just checking no errors are generated.

### Coverage

Coverage is checked with [coverage](https://coverage.readthedocs.io/en/latest/).

To run locally, use
```
coverage run -m unittest
```
and, if the tests pass, view the report with
```
coverage report
```

### Github actions

Whenever changes are made to a branch with an open pull request, tests will be run using GitHub actions.

These are configured in a single workflow **TODO THIS IS BEING UPDATED ATM**

Coverage testing is run, and sent to [codecov.io](https://docs.codecov.io/docs) to generate [online reports](https://app.codecov.io/github/CardiacModelling/pcpostprocess).

## Logging changes

Each PR should add a line (or occasionally multiple lines) to [CHANGELOG.md](./CHANGELOG.md).
This should be a very concise summary of the work done, and link to the PR itself for further info.
Changes are classified as `Added`, `Changed`, `Deprecated`, `Removed`, or `Fixed`.

For example, the first entry in our Changelog was:
```
- Added
  - [#104](https://github.com/CardiacModelling/pcpostprocess/pull/104) Added a CHANGELOG.md and CONTRIBUTING.md
```

Changelog entries are intended for _users_, and so should focus on changes to the public API or command line interface.
Changes to internals are less likely to need reporting.

## Packaging

This project uses a minimal [`setup.py`](./setup.py), and instead uses [pyproject.toml](./pyproject.toml).

### Versioning

Version numbers are not set in the code, but derived from git tags, using [setuptools-scm](https://setuptools-scm.readthedocs.io/en/latest/).
This is run every time setuptools is run, e.g. with
```
pip install -e .[test]
```

Versions numbers take the form `X.Y.Z` where X, Y, and Z are "major", "minor", and "revision" numbers.
Changes to the public interface should be reflected in an updated minor version.
Small changes should be indicated with revisions.
These numbers don't stop at 9, so e.g. 1.11.12 is a viable number.

**The version number is only changed when a new release is made**

### Releases

Releases, like other changes, are made on a branch, using the following procedure:

1. Create a new branch.
2. Update the changelog, replacing the "Unreleased" header with a version number and date, e.g. `## [0.2.0] - 2025-11-11`.
3. Commit the change.
4. Add a tag using e.g. `git tag v0.2.0`, and push it with `git push --tags`.
5. Update the changelog, adding a new "Unreleased" header with empty categories.
6. Merge the PR.
7. [Add a new release](https://github.com/CardiacModelling/pcpostprocess/releases) in GitHub, using the tag you created, and copy in the changes from the changelog.

**TODO: THERE IS NO PACKAGING ON PYPI ATM [#105](https://github.com/CardiacModelling/pcpostprocess/issues/105)**

### Licensing

Licensing information is provided in a separate [LICENSE](./LICENSE) file.

