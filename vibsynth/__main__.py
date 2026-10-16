import sys

from .cli import main

if __name__ == "__main__":          # worker processes import this module: no side effects
    sys.exit(main())
