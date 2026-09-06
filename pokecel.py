#!/usr/bin/env python3
"""Compatibility entry point for the PokeCel viewer."""
from pokecel_viewer import *  # re-export run/main for existing fetch/ZIP commands

if __name__ == "__main__":
    raise SystemExit(main())
