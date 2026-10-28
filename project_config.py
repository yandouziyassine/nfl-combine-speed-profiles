"""Portable project defaults shared by the command-line analysis scripts."""
from __future__ import annotations

import os
from pathlib import Path


# A local file in the working directory is a convenient default. Set
# NFL_BDB_INPUT to an absolute ZIP path or extracted competition directory on
# another machine; command-line --archive always takes precedence.
DEFAULT_ARCHIVE = Path(os.environ.get("NFL_BDB_INPUT", "nfl-big-data-bowl-2027.zip")).expanduser()
