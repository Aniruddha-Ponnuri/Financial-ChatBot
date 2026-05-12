"""Run pytest with a clean plugin environment.

This avoids failures from globally installed pytest plugins (e.g., ROS).
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    args = sys.argv[1:] or ["-q"]
    cmd = [sys.executable, "-m", "pytest", *args]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
