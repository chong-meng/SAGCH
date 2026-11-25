"""Command-line entry point for SAGCH training and evaluation."""
import sys
sys.dont_write_bytecode = True
from sagch_module_ablation.run_ablation import main

if __name__ == "__main__":
    main()
