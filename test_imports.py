import os, sys

# 1. Load treematch
treematch_path = os.path.abspath("treematch")
sys.path.insert(0, treematch_path)
from models.treematch import Trainer
sys.path.remove(treematch_path)

# Clear conflicting top-level modules
for mod in list(sys.modules.keys()):
    if mod == 'models' or mod.startswith('models.') or mod == 'utils' or mod.startswith('utils.'):
        del sys.modules[mod]

# 2. Load AnySat
anysat_path = os.path.abspath("AnySat")
sys.path.append(anysat_path)
from hubconf import AnySat

print("Both imported successfully!")
