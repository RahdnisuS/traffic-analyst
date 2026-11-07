"""Headless traffic simulation core. Nothing in this package knows about rendering or networking."""

from .config import Config, load_config
from .engine import Simulation
from .junction import Junction

__all__ = ["Config", "Junction", "Simulation", "load_config"]
