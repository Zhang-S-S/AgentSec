#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from agentsec.determinism import configure_determinism

configure_determinism()

from agentsec.experiment import main


if __name__ == "__main__":
    main()
