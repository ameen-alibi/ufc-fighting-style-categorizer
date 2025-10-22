import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

scripts = [
    "scripts/01_web_scraping.py",
    "scripts/02_data_cleaning.py",
    "scripts/04_fighters_clustering.py"
]

for script in scripts:
    print(f"Running {script}...")
    os.system(f"python {script}")
