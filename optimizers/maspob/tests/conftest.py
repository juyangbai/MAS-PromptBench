"""Offline MASPOB tests: Hugging Face offline, optional extra site-packages for torch_geometric.

``MASPOB_TEST_SITE_PACKAGES`` may name a directory that holds torch_geometric
(for example an unpacked wheel). It is appended to ``sys.path`` for these tests
only, with bytecode writing disabled so that directory is never modified.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_EXTRA_SITE = os.environ.get("MASPOB_TEST_SITE_PACKAGES")
if _EXTRA_SITE:
    sys.dont_write_bytecode = True
    if _EXTRA_SITE not in sys.path:
        sys.path.append(_EXTRA_SITE)
