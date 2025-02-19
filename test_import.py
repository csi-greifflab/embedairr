import sys
import os

# Add the parent directory to the PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print("PYTHONPATH:", sys.path)

from embedairr.model_selecter import select_model

print("Import successful")
