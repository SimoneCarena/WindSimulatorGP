import os
from pathlib import Path

for file in os.listdir('trajectories'):
    file_name = Path(file).stem
    print(file, file_name)
