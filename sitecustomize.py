"""Test runner guardrails.

Disables auto-loading external pytest plugins that can break test discovery
(e.g., system-installed ROS plugins) while still allowing explicit plugins.
"""

from __future__ import annotations

import os
import sys

if "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in os.environ:
    if any("pytest" in arg for arg in sys.argv):
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
