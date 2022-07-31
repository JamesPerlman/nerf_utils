
import glob
import os
import sys

NGP_DIR = os.environ['NGP_DIR']

sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(NGP_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(NGP_DIR, "build*", "**/*.so"), recursive=True)]
