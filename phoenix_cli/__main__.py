"""Allow running Phoenix CLI as ``python -m phoenix_cli``."""

from .cli import cli

if __name__ == "__main__":
    cli()
