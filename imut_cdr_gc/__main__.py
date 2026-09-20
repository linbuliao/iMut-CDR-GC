"""Support ``python -m imut_cdr_gc`` as an alternative to the installed CLI."""
from .cli import main

raise SystemExit(main())
