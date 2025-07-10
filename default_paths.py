from pathlib import Path

ROOT = Path(__file__).parent
DATA_WILDS_ROOT = (
    "/data/wilds" if Path("/data/wilds").exists() else "/vol/biodata/data/wilds"
)
DATA_IMAGENET = (
    Path("/data/ILSVRC2012")
    if Path("/data/ILSVRC2012").exists()
    else Path("/vol/biodata/data/ILSVRC2012")
)
BREEDS_INFO_DIR = Path("/vol/biodata/data/breeds") / "modified"
DATA_DTD = Path("/vol/biodata/data/DTD-Texture")
