"""Compatibility alias for Mojive's production icon library.

The feasibility workbench and runtime intentionally import the same module so
reviewed geometry cannot drift after promotion.
"""

from mojive.ui import icons as _icons

globals().update({name: getattr(_icons, name) for name in dir(_icons) if not name.startswith("__")})
