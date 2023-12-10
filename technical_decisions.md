# Technical Decisions

This file contains information motivating the technical decisions made in this project. Most of this information can be found in commits' comments, which have been used as discussions in pull requests.

## Dependency management

Dependencies are handled by Poetry.

### Poetry lock file

Poetry developers recommend adding the `poetry.lock` file to version control, as it allows for reproducible builds. For more information on this matter:

- See the [Poetry docs](https://python-poetry.org/docs/basic-usage/#as-an-application-developer)
- See this [Rust Cargo Book extract](https://doc.rust-lang.org/cargo/faq.html#why-have-cargolock-in-version-control) on `Cargo.lock`, which has the same purpose

Additionally, Poetry produces a 'universal' file, as can be seen when going through the file, or as outlined in this [StackOverflow answer by a Poetry maintainer](https://stackoverflow.com/questions/61037557/should-i-commit-lock-file-changes-separately-what-should-i-write-for-the-commi/74045098#74045098). The artifact can be shared between developers working in different environments.

The `poetry.lock` file is kept in sync with the dependencies declaration in `pyproject.toml` using a pre-commit hook.

## Development tooling

- pytest is used for unit testing
- autopep8 is used for autoformating
- flake8 is used for linting

All 3 of them can be run using your IDE, which is recommended.

Autoformating is executed as a pre-commit, because:

- All committed code will be formatted, streamlining parsing and reviewing
- Autoformating does not prevent commits

Pre-commits should not prevent commits as this hinders workflows. Linting and testing are blocking, requiring errors to be fixed. They are better used in a CI pipeline.

### Pre-commit

Pre-commit is a powerful tool to define and share git pre-commit hooks whilst ensuring consistent behaviors.

Files modified by a pre-commit hook are not staged automatically, on purpose ([see this post by the tool's creator](https://stackoverflow.com/questions/64309766/prettier-using-pre-commit-com-does-not-re-stage-changes/64309843#64309843)), ensuring all committed code has been approved by a human.

In this project, the Python version used by pre-commit is pinned, to ensure consistent behavior. Pre-commit creates an isolated environment for each hook [as outlined by the tool's creator](https://stackoverflow.com/a/70780205). By default, it will use the system-installed version of the requested language ([see the docs](https://pre-commit.com/#overriding-language-version)). If pre-commit is installed in a virtual environment, it will use this environment version, so this should not be an issue in general.

## Troubleshooting

Poetry updates can change `poetry.lock` format, rendering it unusable to prior versions. There is currently no straightforward way to enforce a specific Poetry version using the tool alone ([relevant GitHub issue](https://github.com/python-poetry/poetry/issues/3316)). I recommend using the Poetry version outlined in this project's README.