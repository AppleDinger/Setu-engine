import runpy
from pathlib import Path


# Small wrapper — forward execution to the script inside `alias mapping/` so
# `python names_finder.py` keeps working while the authoritative script lives
# in `alias mapping/names_finder.py`.
SCRIPT = Path(__file__).resolve().parent / "alias mapping" / "names_finder.py"
if not SCRIPT.exists():
    print(f"Error: expected script at {SCRIPT} not found.")
    raise SystemExit(1)

runpy.run_path(str(SCRIPT), run_name="__main__")
    
# Wrapper completed — execution forwarded to alias mapping script above.