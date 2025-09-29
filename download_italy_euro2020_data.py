import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from tqdm import tqdm

statsbomb_git_path = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
out_dir = Path("./data/italy_euro2020")
out_dir.mkdir(parents=True, exist_ok=True)

competition_id = 55
season_id = 43
match_ids = [
    3788741,  # Group stage: Italy vs Turkey
    3788754,  # Group stage: Italy vs Switzerland
    3788766,  # Group stage: Italy vs Wales
    3794679,  # Round of 16: Italy vs Austria
    3794686,  # Quarterfinal: Italy vs Belgium
    3795220,  # Semifinal: Italy vs Spain
    3795506,  # Final: Italy vs England
]

# aux function to download files related to particular ids
def download(url, out_path):
    try:
        text = urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"})).read().decode("utf-8")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return True
    except Exception as e:      #avoids crash if 360 data is missing
        print(f"[WARN] Could not fetch {url} -> {e}")
        return False


download(f"{statsbomb_git_path}/competitions.json", out_dir / "competitions.json")
download(f"{statsbomb_git_path}/matches/{competition_id}/{season_id}.json", out_dir / f"matches/{competition_id}/{season_id}.json")

for id in tqdm(match_ids, desc="Downloading Italy Euro2020 files"):
    download(f"{statsbomb_git_path}/events/{id}.json", out_dir / f"events/{id}.json")
    download(f"{statsbomb_git_path}/lineups/{id}.json", out_dir / f"lineups/{id}.json")
    download(f"{statsbomb_git_path}/three-sixty/{id}.json", out_dir / f"three-sixty/{id}.json")
    time.sleep(0.1)

print("Files correctly downloaded in", out_dir, "maintaining StatsBomb repo folder design")
