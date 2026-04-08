"""NY House Price prediction package.

Avoid importing submodules at package import time. Importing `app` here
causes a module to be present in `sys.modules` before the CLI module is
executed with `python -m`, which triggers a runtime warning. Keep package
initialization lightweight.
"""

__all__ = []
