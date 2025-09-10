# main.py
"""
Thin wrapper entry point for Local Learning Coefficient Analysis.
The actual CLI logic has been moved to llc.cli for better modularity.
"""
from llc.cli import main

if __name__ == "__main__":
    main()