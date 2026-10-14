"""Backward-compatible entry point; implementation is in a functional package."""
from _compat import dispatch

dispatch(__name__, __file__)
