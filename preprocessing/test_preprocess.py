import os
from utils import ROOT_DIR

pyfile = os.path.join(ROOT_DIR, 'preprocessing', 'process_raw.py')
print(pyfile)
assert os.path.exists(pyfile), "Preprocessing step not implemented!"