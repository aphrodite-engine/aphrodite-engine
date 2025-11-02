import sys

if __name__ == "__main__":
    print("""DEPRECATED: This script has been moved to the Aphrodite CLI.

Please use the following command instead:
    aphrodite bench throughput

For help with the new command, run:
    aphrodite bench throughput --help

Alternatively, you can run the new command directly with:
    python -m aphrodite.entrypoints.cli.main bench throughput --help
""")
    sys.exit(1)
