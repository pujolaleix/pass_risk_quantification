import json
from pathlib import Path

def file_exists(dir_path: Path):
    return dir_path.exists() and dir_path.is_file()

def count_freeze_frames(id: int, data_dir: Path) -> int:
    p = data_dir / "three-sixty" / f"{id}.json"
    if not p.exists():
        return 0
    frames = json.loads(p.read_text(encoding="utf-8"))
    return sum(1 for fr in frames if fr.get("freeze_frame"))