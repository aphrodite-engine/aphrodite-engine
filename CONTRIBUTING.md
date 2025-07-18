## Contributing Guidelines for Aphrodite Engine

Aphrodite Engine welcomes any and all contributors to submit fixes, features, issues - no matter the scope.

### Code of Ethics
If you wish, you can read the [Aphrodite Engine Code of Conduct](./CODE_OF_CONDUCT.md). You don't need to follow it! :)

### Setup
For development, you will need to build the engine from source.

```sh
pip install -e .
pip install -r requirements/dev.txt
```

Make sure you build in editable mode with `-e`.

Our test units at the moment are incomplete, so you can skip test units for now.

After submitting a patch, make sure to run the linter/formatter before asking for a review:

```sh
sh ./formatting.sh
```

Thank you for your interest in Aphrodite Engine!