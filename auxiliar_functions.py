import json
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import matplotlib as mpl
from mplsoccer import Pitch
import os

# -------------------------------------
# ----- Data Processing functions -----
# -------------------------------------

def file_exists(dir_path: Path):
    return dir_path.exists() and dir_path.is_file()


def count_freeze_frames(id: int, data_dir: Path) -> int:
    p = data_dir / "three-sixty" / f"{id}.json"
    if not p.exists():
        return 0
    frames = json.loads(p.read_text(encoding="utf-8"))
    return sum(1 for fr in frames if fr.get("freeze_frame"))



# --------------------------------------
# ----- Data Engineering functions -----
# --------------------------------------

def load_events(id: int, data_dir: Path):
    p = data_dir/"events"/f"{id}.json"
    return json.loads(p.read_text(encoding="utf-8"))

def load_lineups(id: int, data_dir: Path):
    p = data_dir/"lineups"/f"{id}.json"
    lineup_json = json.loads(p.read_text(encoding="utf-8"))
    
    players = []
    for team in lineup_json:
        team_name = team["team_name"]
        for pl in team["lineup"]:
            players.append({
                "match_id": id,
                "team": team_name,
                "player_name": pl["player_name"],
                "jersey_number": pl["jersey_number"],
            })
    return pd.DataFrame(players)


def load_360(id: int, data_dir: Path):
    p = data_dir/"three-sixty"/f"{id}.json"
    frames_json = json.loads(p.read_text(encoding="utf-8"))

    rows = []
    for fr in frames_json:
        ei = fr.get("event_uuid")
        for pl in fr.get("freeze_frame", []) or []:
            loc = pl.get("location",[None,None])
            rows.append({
                "event_index": ei,
                "teammate": bool(pl.get("teammate", False)),
                "x": loc[0], "y": loc[1]
            })
    return pd.DataFrame(rows)



def third_x(x):
    # StatsBomb pitch: [0,120]
    if x < 40:  return "defensive"
    if x < 80:  return "middle"
    return "attacking"


def pass_features_from_events(events_json, features) -> pd.DataFrame:
    df = pd.json_normalize(events_json)
    df = df[df["type.name"] == "Pass"].copy()

    #feature engineering
    df["time"] = df["minute"].astype(int).astype(str).str.zfill(2) + ":" + df["second"].astype(int).astype(str).str.zfill(2)
    df["team"]   = df["team.name"]
    df["player_name"] = df["player.name"]
    df["recipient"] = df["pass.recipient.name"]
    df["outcome"] = df["pass.outcome.name"].fillna("Complete")
    df["is_complete"] = df["outcome"].eq("Complete")
    df["sx"] = df["location"].str[0]
    df["sy"] = df["location"].str[1]
    df["ex"] = df["pass.end_location"].str[0]
    df["ey"] = df["pass.end_location"].str[1]
    df["dx"] = df["ex"] - df["sx"]
    df["dy"] = df["ey"] - df["sy"]
    df["length"] = np.hypot(df["dx"], df["dy"])
    df["angle"]  = np.arctan2(df["dy"], df["dx"])
    df["direction"] = np.where(df["dx"] > 10, "forward", np.where(df["dx"] < -10, "backward", "lateral"))
    df["is_progressive"] = df["dx"] > 20
    df["height"] = df["pass.height.name"].fillna("Unknown")
    df["is_cross"] = df["pass.cross"].fillna(False)
    df["is_through"] = df["pass.through_ball"].fillna(False)
    df["zone_origin"] = df["sx"].apply(third_x)
    df["zone_dest"]   = df["ex"].apply(third_x)

    return df[features]


def pressure_metrics(df: pd.DataFrame, pressure_thr: float, bypass_margin_thr: float):
    sx = float(df["sx"].iloc[0])
    sy = float(df["sy"].iloc[0])
    ex = float(df["ex"].iloc[0])
    ey = float(df["ey"].iloc[0])

    # opponents only
    opponents = df.loc[df["teammate"]==False, ["x", "y"]].dropna()

    if opponents.empty:
        return pd.Series({
            "passer_pressure": 0,
            "passer_pressure_dist": np.nan,
            "receiver_pressure": 0,
            "receiver_pressure_dist": np.nan,
            "bypassed_opponents": 0,
        })

    ox = opponents["x"].to_numpy()
    oy = opponents["y"].to_numpy()

    d_passer   = np.hypot(ox - sx, oy - sy)         # distances between pass start position and in-frame opponents
    d_receiver = np.hypot(ox - ex, oy - ey)         # distances between pass end position and in-frame opponents

    #bypassed opponents logic
    vx = ex - sx
    vy = ey - sy
    denom = vx*vx + vy*vy
    t = ((ox-sx)*vx + (oy-sy)*vy) / denom
    on_seg = (t>0.0) & (t<1.0)
    projx = sx + t*vx
    projy = sy + t*vy
    perp = np.hypot(ox - projx, oy - projy)
    bypassed_opp = int((on_seg & (perp<=bypass_margin_thr)).sum())
    if denom <= 1e-9:
        bypassed_opp = 0

    return pd.Series({
        "passer_pressure": int((d_passer<=pressure_thr).sum()),
        "passer_pressure_dist": float(d_passer.min()),
        "receiver_pressure": int((d_receiver<=pressure_thr).sum()),
        "receiver_pressure_dist": float(d_receiver.min()),
        "bypassed_opponents": bypassed_opp
    })


# --------------------------------------
# ---- Data Visualization functions ----
# --------------------------------------

def per_field_zone_visual(vcol, nx, ny, team_color, df, save_dir):

    xcol = 'sx'
    ycol = 'sy'

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#2C2C2C", line_color='white', linewidth=1.5)
    fig, ax = pitch.draw(figsize=(7,5))

    # compute per-team metric grid
    team_grids = {}
    for team, g in df.groupby("team"):
        s = pitch.bin_statistic(
            g[xcol].to_numpy(), g[ycol].to_numpy(),
            values=g[vcol].to_numpy(),
            statistic="mean", bins=(nx, ny)
        )
        team_grids[team] = pd.DataFrame(s["statistic"], index=range(ny), columns=range(nx))

    # Compute bin center to display text
    pitch_length, pitch_width = pitch.dim.length, pitch.dim.width # [120, 80] in StatsBomb pitch 
    x_edges = np.linspace(0, pitch_length, nx + 1)
    y_edges = np.linspace(0, pitch_width, ny + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    # small vertical offsets so multiple team numbers don't overlap in a cell
    nT = max(1, len(df['team'].unique()))
    cell_h = y_edges[1] - y_edges[0]
    offsets = {t: ((i - (nT - 1) / 2) * (cell_h * 0.20)) for i, t in enumerate(df['team'].unique())}


    # display soft white grid lines to split field in areas
    for xe in x_edges[1:-1]:
        ax.plot([xe, xe], [0, pitch_width], color="white", alpha=0.25, lw=1.2, zorder=5)
    for ye in y_edges[1:-1]:
        ax.plot([0, pitch_length], [ye, ye], color="white", alpha=0.25, lw=1.2, zorder=5)
    

    # write each team's mean in its color at each bin center
    for t in df['team'].unique():
        grid = team_grids[t]
        dy = offsets[t]
        for iy in range(ny):
            for ix in range(nx):
                val = grid.iloc[iy, ix]
                if np.isfinite(val):
                    ax.text(
                        x_centers[ix], y_centers[iy] + dy, f"{val:.2f}",
                        ha="center", va="center",
                        color=team_color[t], fontsize=10, fontweight="bold", zorder=6,
                        path_effects=[pe.withStroke(linewidth=2, foreground="black", alpha=0.6)]
                    )

    # title
    ax.set_title(f"{vcol} average by field zone",
                fontsize=10, color="black", pad=15, fontweight="bold")
    
    # arrow that shows attacking direction
    arrow = FancyArrowPatch(
        (15, 75), (105, 75),
        arrowstyle='-|>', mutation_scale=25,
        color="white", lw=2, alpha=0.55
    )
    ax.add_patch(arrow)
    ax.text(60, 80, "Attacking direction", ha="center", va="bottom", color="white", alpha=0.55, fontsize=8, fontweight="bold")

    # save as .png
    os.makedirs(save_dir, exist_ok=True)
    fname = f"per_zone_{vcol}_{nx}x{ny}.png"
    save_path = os.path.join(save_dir, fname)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())



def draw_numbered_circle(ax, x, y, number, facecolor, size=220, text_color="white", lw=1.2):
    sc = ax.scatter(x, y, s=size, marker="o", facecolor=facecolor,
               edgecolor="black", linewidths=lw, zorder=6)
    txt = ax.text(x, y, "" if pd.isna(number) else f"{int(number)}",
                  ha="center", va="center", fontsize=9, color=text_color,
                  weight="bold", zorder=7)
    
    txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="black")])
    
    return [sc, txt]


def update_player_markers(player_markers, player_name, x, y, circle, number):
    if pd.isna(player_name):
        return
    pid = str(player_name)
    label = "" if pd.isna(number) else f"{int(number)}"

    if pid in player_markers:
        sc = player_markers[pid]['circle']
        tx = player_markers[pid]['text']
        try:
            sc.remove()
            tx.remove()
        except Exception:
            pass
        new_sc, new_tx = circle
        player_markers[pid] = {'circle': new_sc, 'text': new_tx}

        sc.set_offsets([[x, y]])
        tx.set_position((x, y))
        tx.set_text(label)
    else:
        sc, tx = circle
        player_markers[pid] = {'circle': sc, 'text': tx}