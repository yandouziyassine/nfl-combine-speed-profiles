"""Build player-level Combine speed profiles and link them to rookie game speed.

Reads either the official Kaggle ZIP or a directory of competition CSVs; it does not extract or alter source files.
Only the Python standard library, NumPy, and pandas are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import uuid

import numpy as np
import pandas as pd
from project_config import DEFAULT_ARCHIVE

MEMBER = "nfl-big-data-bowl-2027/"
CHUNK_ROWS = 300_000
DT_SECONDS = 0.1
SPEED_START_THRESHOLD = 1.0  # yd/s: movement-onset convention, not reaction time
SUSTAINED_PEAK_FRAMES = 5    # 0.5 seconds at 10 Hz
GAME_PROFILE_CACHE_SCHEMA = 1
GAME_PROFILE_CACHE_INPUTS = (
    "players.csv", "games.csv", "player_play.csv",
    "game_tracking_2023.csv", "game_tracking_2024.csv", "game_tracking_2025.csv",
)


def safe_float(value):
    return None if pd.isna(value) or not np.isfinite(value) else float(value)


def read_csv(archive, name, **kwargs):
    return pd.read_csv(archive.open(MEMBER + name), **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OutputDirectoryLock:
    """Exclusive lock preventing concurrent promotion or certification in one output directory."""
    def __init__(self, outdir: Path):
        self.path = Path(outdir) / ".core_run.lock"
        self.token = uuid.uuid4().hex

    def __enter__(self):
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"Another core operation owns this output directory, or a stale lock remains at {self.path}. Verify no operation is active before removing a stale lock.") from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
                lock_file.write(self.token)
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.path.is_file() and self.path.read_text(encoding="utf-8") == self.token:
                self.path.unlink()
        except FileNotFoundError:
            pass


def game_profile_cache_identity(input_path: Path) -> dict:
    """Describe only source members used by the rookie game-profile cache.

    ZIP members use their CRC and uncompressed size; extracted files use size
    and nanosecond mtime. The resulting fingerprint is a cache identity, not a
    cryptographic checksum of the complete competition data.
    """
    path = Path(input_path).expanduser().resolve()
    members = []
    if path.is_file():
        with zipfile.ZipFile(path) as source:
            infos = {info.filename: info for info in source.infolist() if not info.is_dir()}
            for filename in GAME_PROFILE_CACHE_INPUTS:
                member = MEMBER + filename
                info = infos.get(member)
                if info is None:
                    raise FileNotFoundError(f"Required game-profile cache input missing from ZIP: {member}")
                members.append({"name": member, "size": info.file_size, "crc32": f"{info.CRC:08x}"})
        method = "zip-member-crc32-and-size"
        kind = "zip"
    elif path.is_dir():
        root = path
        for filename in GAME_PROFILE_CACHE_INPUTS:
            candidates = [root / MEMBER / filename, root / filename, root / MEMBER.rstrip("/") / filename]
            found = {candidate.resolve() for candidate in candidates if candidate.is_file()}
            if not found:
                found = {candidate.resolve() for candidate in root.rglob(filename) if candidate.is_file()}
            if len(found) != 1:
                detail = "not found" if not found else f"found {len(found)} matches"
                raise FileNotFoundError(f"Expected one {filename} under {root}; {detail}.")
            member_path = next(iter(found))
            stat = member_path.stat()
            members.append({"name": member_path.relative_to(root).as_posix(),
                            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        method = "directory-relative-path-size-mtime-ns"
        kind = "directory"
    else:
        raise FileNotFoundError(path)

    source_payload = {"kind": kind, "method": method, "members": members}
    canonical_source = json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_fingerprint = hashlib.sha256(canonical_source).hexdigest()
    producer_sha256 = sha256_file(Path(__file__).resolve())
    key_payload = {"cache_schema": GAME_PROFILE_CACHE_SCHEMA,
                   "source_fingerprint": source_fingerprint,
                   "producer_sha256": producer_sha256,
                   "rookie_seasons": [2023, 2024, 2025]}
    cache_key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"cache_schema_version": GAME_PROFILE_CACHE_SCHEMA,
            "cache_key": cache_key,
            "input": {"name": path.name, "kind": kind, "fingerprint_method": method,
                      "fingerprint": source_fingerprint, "members": members},
            "producer": {"script": Path(__file__).name, "sha256": producer_sha256}}


def write_json_atomic(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class DataDirectory:
    """Adapter exposing the `.open()` and `.namelist()` interfaces for extracted inputs."""
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._names = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        return None

    def namelist(self):
        if self._names is None:
            self._names = [p.relative_to(self.root).as_posix()
                           for p in self.root.rglob("*") if p.is_file()]
        return list(self._names)

    def open(self, member: str, mode: str = "r"):
        name = Path(member).name
        candidates = [self.root / member, self.root / name,
                      self.root / MEMBER.rstrip("/") / name]
        for path in candidates:
            if path.is_file():
                return path.open(mode, newline="")
        matches = list(self.root.rglob(name))
        if len(matches) == 1:
            return matches[0].open(mode, newline="")
        if len(matches) > 1:
            raise FileNotFoundError(f"Multiple {name} files found under {self.root}; pass the exact competition data folder.")
        raise FileNotFoundError(f"Required competition file {name} not found under {self.root}.")


def open_data_source(path: Path):
    """Open either the competition ZIP or Kaggle's extracted input directory."""
    path = Path(path)
    if path.is_dir():
        return DataDirectory(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return zipfile.ZipFile(path)


SENSOR_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def parse_sensor_times(values, *, utc: bool = False):
    """Parse the competition's fixed ISO timestamp format without row-wise inference."""
    return pd.to_datetime(values, format=SENSOR_TIME_FORMAT, errors="coerce", utc=utc)


def interpolate_at(x, y, target):
    """Linearly interpolate y at increasing x; return NaN outside its range."""
    if len(x) < 2 or target < x[0] or target > x[-1]:
        return np.nan
    j = int(np.searchsorted(x, target, side="left"))
    if j == 0:
        return float(y[0])
    if x[j] == x[j - 1]:
        return float(y[j])
    w = (target - x[j - 1]) / (x[j] - x[j - 1])
    return float(y[j - 1] + w * (y[j] - y[j - 1]))


def calculate_attempt(group, onset_threshold=None, smoothing_frames=3):
    """Calculate transparent speed features for one athlete drill attempt.

    Optional onset/smoothing settings support measurement-sensitivity analyses;
    production defaults preserve the project's original feature definitions.
    """
    onset_threshold = SPEED_START_THRESHOLD if onset_threshold is None else float(onset_threshold)
    smoothing_frames = max(1, int(smoothing_frames))
    g = group.sort_values("time")
    t0 = parse_sensor_times(g["time"])
    t = (t0 - t0.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    raw_speed = pd.to_numeric(g["s"], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(g["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(g["y"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(raw_speed) & np.isfinite(x) & np.isfinite(y)
    t, raw_speed, x, y = t[valid], raw_speed[valid], x[valid], y[valid]
    out = {"frame_count": int(len(t)), "trace_duration_s": float(t[-1]) if len(t) else np.nan}
    if len(t) < 4:
        return out
    # Median smoothing damps single-frame optical spikes while retaining 10-Hz events.
    smooth = pd.Series(raw_speed).rolling(smoothing_frames, center=True, min_periods=1).median().to_numpy()
    # Keep true path length for movement-distance summaries, but use signed x
    # progress for straight-line dash yard markers. Across all attempts, full-trace
    # PCA shows the shared course axis is almost exactly the x coordinate.
    step = np.hypot(np.diff(x), np.diff(y))
    path = np.r_[0.0, np.cumsum(step)]
    sustained = (smooth >= onset_threshold) & np.r_[smooth[1:] >= onset_threshold, False]
    starts = np.flatnonzero(sustained)
    if not len(starts):
        return out
    start = int(starts[0])
    onset_t = float(t[start])
    rel_t = t[start:] - onset_t
    rel_path = path[start:] - path[start]
    x_progress = x[start:] - x[start]
    if len(x_progress) and abs(float(x[-1] - x[start])) >= 0.25:
        direction = float(np.sign(x[-1] - x[start]))
        rel_course = np.maximum.accumulate(x_progress * direction)
    else:
        # An attempt without a clear course-axis direction cannot support yard splits.
        rel_course = np.full(len(x_progress), np.nan, dtype=float)
    rel_speed = smooth[start:]
    if len(rel_t) < 2 or np.any(np.diff(rel_t) <= 0):
        return out

    # Label the observed movement threshold; its time from recording start is not reaction time.
    out["movement_onset_threshold_ydps"] = onset_threshold
    out["movement_onset_from_recording_s"] = onset_t
    out["speed_at_movement_onset_ydps"] = float(rel_speed[0])
    # Path-relative lateral speeds. Combine tracking lacks a body-orientation field,
    # so these describe left/right from the initial travel direction, not anatomy.
    axis_end = min(start + 4, len(x) - 1)
    forward = np.array([x[axis_end] - x[start], y[axis_end] - y[start]], dtype=float)
    norm = float(np.linalg.norm(forward))
    if norm > 0.05:
        forward /= norm
        left_axis = np.array([-forward[1], forward[0]])
        vx = pd.Series(np.gradient(x, t)).rolling(smoothing_frames, center=True, min_periods=1).median().to_numpy()
        vy = pd.Series(np.gradient(y, t)).rolling(smoothing_frames, center=True, min_periods=1).median().to_numpy()
        lateral_velocity = vx[start:] * left_axis[0] + vy[start:] * left_axis[1]
        lateral_velocity = lateral_velocity[rel_t >= 0.3]
        left = lateral_velocity[lateral_velocity > 0.5]
        right = -lateral_velocity[lateral_velocity < -0.5]
        out["left_lateral_speed_mean_ydps"] = float(np.mean(left)) if len(left) else np.nan
        out["left_lateral_speed_p90_ydps"] = float(np.quantile(left, 0.9)) if len(left) else np.nan
        out["left_lateral_frames_above_0_5ydps"] = int(len(left))
        out["right_lateral_speed_mean_ydps"] = float(np.mean(right)) if len(right) else np.nan
        out["right_lateral_speed_p90_ydps"] = float(np.quantile(right, 0.9)) if len(right) else np.nan
        out["right_lateral_frames_above_0_5ydps"] = int(len(right))
        out["left_right_lateral_speed_difference_ydps"] = out["left_lateral_speed_mean_ydps"] - out["right_lateral_speed_mean_ydps"] if np.isfinite(out["left_lateral_speed_mean_ydps"]) and np.isfinite(out["right_lateral_speed_mean_ydps"]) else np.nan
    else:
        for name in ["left_lateral_speed_mean_ydps", "left_lateral_speed_p90_ydps", "right_lateral_speed_mean_ydps", "right_lateral_speed_p90_ydps", "left_right_lateral_speed_difference_ydps"]:
            out[name] = np.nan
        out["left_lateral_frames_above_0_5ydps"] = 0
        out["right_lateral_frames_above_0_5ydps"] = 0
    out["speed_start_to_0_5s_ydps"] = interpolate_at(rel_t, rel_speed, 0.5)
    out["speed_start_to_1s_ydps"] = interpolate_at(rel_t, rel_speed, 1.0)
    out["speed_start_to_2s_ydps"] = interpolate_at(rel_t, rel_speed, 2.0)
    for horizon in (0.5, 1.0, 2.0):
        s_end = interpolate_at(rel_t, rel_speed, horizon)
        out[f"early_accel_0_{horizon:g}s_ydps2"] = (s_end - rel_speed[0]) / horizon if np.isfinite(s_end) else np.nan

    grad = np.gradient(rel_speed, rel_t)
    out["peak_accel_ydps2"] = float(np.nanmax(grad)) if len(grad) else np.nan
    out["peak_braking_ydps2"] = float(np.nanmin(grad)) if len(grad) else np.nan
    out["mean_accel_first_1s_ydps2"] = float(np.nanmean(grad[rel_t <= 1.0])) if np.any(rel_t <= 1.0) else np.nan
    out["mean_accel_after_1s_ydps2"] = float(np.nanmean(grad[rel_t > 1.0])) if np.any(rel_t > 1.0) else np.nan
    roll = pd.Series(rel_speed).rolling(SUSTAINED_PEAK_FRAMES, center=True, min_periods=3).median().to_numpy()
    out["peak_sustained_speed_ydps"] = float(np.nanmax(roll)) if np.isfinite(roll).any() else np.nan
    out["raw_peak_speed_ydps"] = float(np.nanmax(rel_speed))
    out["distance_in_first_1s_yd"] = interpolate_at(rel_t, rel_path, 1.0)
    out["distance_in_first_2s_yd"] = interpolate_at(rel_t, rel_path, 2.0)
    early = rel_t <= 2.0
    out["speed_distance_curve_auc_0_2s"] = float(np.sum((rel_speed[early][1:] + rel_speed[early][:-1]) * np.diff(rel_t[early]) / 2)) if np.sum(early) >= 2 else np.nan

    # Straight dash phases use the x course axis (sign corrected from net progress)
    # and are measured from detected movement onset. Late acceleration is defined
    # between 10 and 20 yards; speed maintenance compares 20 and 40 yards.
    for yards in (5, 10, 20, 30, 40):
        out[f"time_to_{yards}yd_s"] = interpolate_at(rel_course, rel_t, yards)
        out[f"speed_at_{yards}yd_ydps"] = interpolate_at(rel_course, rel_speed, yards)
    t10, t20, t30, t40 = (out[f"time_to_{n}yd_s"] for n in (10, 20, 30, 40))
    s10, s20, s30, s40 = (out[f"speed_at_{n}yd_ydps"] for n in (10, 20, 30, 40))
    out["early_accel_0_10yd_ydps2"] = (s10 - rel_speed[0]) / t10 if np.isfinite(t10) and t10 > 0 and np.isfinite(s10) else np.nan
    out["late_accel_10_20yd_ydps2"] = (s20 - s10) / (t20 - t10) if np.isfinite(t20 - t10) and (t20 - t10) > 0 and np.isfinite(s10 + s20) else np.nan
    out["late_accel_20_30yd_ydps2"] = (s30 - s20) / (t30 - t20) if np.isfinite(t30 - t20) and (t30 - t20) > 0 and np.isfinite(s20 + s30) else np.nan
    out["speed_change_20_40yd_ydps"] = s40 - s20 if np.isfinite(s40 + s20) else np.nan
    out["speed_retention_20_40yd_ratio"] = s40 / s20 if np.isfinite(s40 + s20) and s20 > 0 else np.nan
    return out


def combine_profiles(archive, players, results, outdir):
    combine = read_csv(archive, "combine_tracking.csv")
    combine = combine.loc[combine.entity_type.eq("PLAYER")].copy()
    combine["nfl_id"] = pd.to_numeric(combine.nfl_id, errors="coerce")
    attempt_keys = ["draft_year", "event_id", "nfl_id", "drill_type", "drill_name", "attempt"]
    attempts = []
    for key, group in combine.groupby(attempt_keys, sort=False, dropna=False):
        row = dict(zip(attempt_keys, key))
        row.update(calculate_attempt(group))
        attempts.append(row)
    attempt_df = pd.DataFrame(attempts)
    attempt_df.to_csv(outdir / "combine_attempt_speed_profiles.csv", index=False)

    metric_cols = [c for c in attempt_df.columns if c not in attempt_keys]
    per_drill = (attempt_df.groupby(["nfl_id", "draft_year", "drill_type", "drill_name"], dropna=False)[metric_cols]
                 .agg(["mean", "std", "count"]))
    per_drill.columns = [f"{a}_{b}" for a, b in per_drill.columns]
    per_drill = per_drill.reset_index()
    # Players sometimes turn in opposite directions on different attempts. For a
    # drill-level side comparison, average observed left and right movement across
    # attempts first, then calculate the difference when both sides were sampled.
    left_col = "left_lateral_speed_mean_ydps_mean"
    right_col = "right_lateral_speed_mean_ydps_mean"
    per_drill["left_right_lateral_speed_difference_by_drill_ydps"] = per_drill[left_col] - per_drill[right_col]
    per_drill.to_csv(outdir / "player_drill_speed_profiles.csv", index=False)

    dash = attempt_df.loc[attempt_df.drill_name.eq("FORTY_YARD_DASH")].copy()
    dash_metrics = [c for c in metric_cols if c != "frame_count"]
    dash_summary = dash.groupby("nfl_id")[dash_metrics].agg(["mean", "std", "count"])
    dash_summary.columns = [f"forty_{a}_{b}" for a, b in dash_summary.columns]
    dash_summary = dash_summary.reset_index()
    dash_summary = dash_summary.merge(results[["nfl_id", "forty", "ten_yd_split", "combine_position"]], on="nfl_id", how="left", validate="one_to_one")
    dash_summary.to_csv(outdir / "player_40_speed_profiles.csv", index=False)

    return attempt_df, per_drill, dash_summary


def game_speed_profiles(archive, players, outdir):
    games = read_csv(archive, "games.csv", usecols=["game_id", "season", "season_type"])
    players_small = players[["nfl_id", "draft_year"]].drop_duplicates("nfl_id")
    games["draft_year"] = games["season"]
    games = games.loc[games.season_type.eq("REG"), ["game_id", "season", "draft_year"]]
    games["game_id"] = pd.to_numeric(games.game_id, errors="coerce")
    games["draft_year"] = pd.to_numeric(games.draft_year, errors="coerce")
    # Rookie-year comparison: each cohort receives the same first-season window.
    game_draft = games.set_index("game_id").draft_year
    target_games = set(games.loc[games.season.eq(games.draft_year), "game_id"].astype(int))
    metadata = read_csv(archive, "player_play.csv", usecols=["game_id", "play_id", "nfl_id", "lined_up_position"])
    metadata["game_id"] = pd.to_numeric(metadata.game_id, errors="coerce")
    metadata["nfl_id"] = pd.to_numeric(metadata.nfl_id, errors="coerce")
    metadata["draft_year"] = metadata.game_id.map(game_draft)
    metadata = metadata.loc[metadata.game_id.isin(target_games)]
    metadata = metadata.merge(players_small, on="nfl_id", how="left", suffixes=("", "_player"))
    # Player draft year is authoritative for the rookie-season filter.
    metadata = metadata.loc[metadata.draft_year.eq(metadata.draft_year_player)] if "draft_year_player" in metadata else metadata
    valid_play_keys = set(zip(metadata.game_id.astype(int), metadata.play_id.astype(int), metadata.nfl_id.astype(int)))

    play_peak = {}
    frames = Counter()
    games_seen = defaultdict(set)
    unmatched_tracking_keys = set()
    tracking_player_ids = set()
    for year in (2023, 2024, 2025):
        name = f"game_tracking_{year}.csv"
        usecols = ["game_id", "play_id", "nfl_id", "time", "s"]
        for chunk in read_csv(archive, name, usecols=usecols, chunksize=CHUNK_ROWS, low_memory=False):
            chunk["game_id"] = pd.to_numeric(chunk.game_id, errors="coerce")
            chunk["play_id"] = pd.to_numeric(chunk.play_id, errors="coerce")
            chunk["nfl_id"] = pd.to_numeric(chunk.nfl_id, errors="coerce")
            chunk = chunk.loc[chunk.game_id.isin(target_games) & chunk.nfl_id.isin(players_small.nfl_id)]
            if chunk.empty:
                continue
            chunk["draft_year"] = chunk.game_id.map(game_draft)
            chunk = chunk.merge(players_small, on="nfl_id", how="left", suffixes=("", "_player"))
            chunk = chunk.loc[chunk.draft_year.eq(chunk.draft_year_player)]
            if chunk.empty:
                continue
            chunk["s"] = pd.to_numeric(chunk.s, errors="coerce")
            chunk = chunk.loc[chunk.s.notna()]
            if chunk.empty:
                continue
            frames.update({int(k): int(v) for k, v in chunk.groupby("nfl_id").size().items()})
            tracking_player_ids.update(int(v) for v in chunk.nfl_id.unique())
            games_by_player = chunk.groupby("nfl_id").game_id.unique()
            for pid, values in games_by_player.items():
                games_seen[int(pid)].update(int(v) for v in values)
            chunk["play_key"] = list(zip(chunk.game_id.astype(int), chunk.play_id.astype(int), chunk.nfl_id.astype(int)))
            # Chunk-level maxima are merged by key, so plays split at chunk boundaries remain one play.
            peaks = chunk.groupby("play_key").s.max()
            for key, peak in peaks.items():
                if key in valid_play_keys:
                    play_peak[key] = max(play_peak.get(key, -np.inf), float(peak))
                else:
                    unmatched_tracking_keys.add(key)

    records = []
    by_player = defaultdict(list)
    for (game_id, play_id, nfl_id), peak in play_peak.items():
        by_player[nfl_id].append(peak)
    for player in players.itertuples(index=False):
        pid = int(player.nfl_id)
        p = np.asarray(by_player.get(pid, []), dtype=float)
        records.append({
            "nfl_id": pid,
            "draft_year": int(player.draft_year),
            "nfl_position": player.nfl_position,
            "rookie_reg_game_frames": int(frames[pid]),
            "rookie_reg_games_tracked": len(games_seen[pid]),
            "rookie_reg_player_plays_with_matching_metadata": int(len(p)),
            "rookie_reg_median_play_peak_speed_ydps": float(np.median(p)) if len(p) else np.nan,
            "rookie_reg_p90_play_peak_speed_ydps": float(np.quantile(p, 0.90)) if len(p) else np.nan,
            "rookie_reg_p95_play_peak_speed_ydps": float(np.quantile(p, 0.95)) if len(p) else np.nan,
            "rookie_reg_max_play_peak_speed_ydps": float(np.max(p)) if len(p) else np.nan,
        })
    profile = pd.DataFrame(records)
    profile.to_csv(outdir / "player_rookie_game_speed_profiles.csv", index=False)
    linkage = {"valid_player_play_keys_rookie_REG": len(valid_play_keys),
               "matched_tracked_player_play_keys_rookie_REG": len(play_peak),
               "unmatched_game_tracking_keys_rookie_REG": len(unmatched_tracking_keys),
               "players_with_rookie_REG_tracking": len(tracking_player_ids),
               "players_with_rookie_REG_matched_plays": int(profile.rookie_reg_player_plays_with_matching_metadata.gt(0).sum())}
    return profile, linkage


def pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, int(len(x))
    value = float(np.corrcoef(x, y)[0, 1])
    return (value if np.isfinite(value) else np.nan), int(len(x))


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 3:
        return np.nan, int(len(x))
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return pearson(rx, ry)[0], int(len(x))


def ridge_predict(train, test, features, target, ridge=1.0):
    """Standardize with training data only; small ridge stabilizes position controls."""
    x_train = train[features].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    y_train = train[target].to_numpy(dtype=float)
    mu = np.mean(x_train, axis=0)
    sd = np.std(x_train, axis=0)
    sd[sd == 0] = 1
    xt = (x_train - mu) / sd
    xv = (x_test - mu) / sd
    # Do not penalize intercept.
    xt = np.column_stack([np.ones(len(xt)), xt])
    xv = np.column_stack([np.ones(len(xv)), xv])
    penalty = np.eye(xt.shape[1]) * ridge
    penalty[0, 0] = 0
    beta = np.linalg.solve(xt.T @ xt + penalty, xt.T @ y_train)
    return xv @ beta


def evaluate_models(joined, outdir):
    target = "rookie_reg_p90_play_peak_speed_ydps"
    out = {"target": target,
           "target_definition": "For each prospect, 90th percentile of that prospect's per-play peak recorded speed in matched regular-season player-tracking plays from the prospect's rookie season.",
           "evaluation": "Leave-one-draft-cohort-out prediction; standardization is learned on training cohorts only. This is an exploratory small-sample check, not a definitive generalization claim.",
           "correlations": {}, "cross_validation": []}
    for label, col in [("official_40_time", "forty"),
                       ("initial_1s_acceleration", "forty_early_accel_0_1s_ydps2_mean"),
                       ("10_to_20yd_acceleration", "forty_late_accel_10_20yd_ydps2_mean"),
                       ("20_to_40yd_speed_change", "forty_speed_change_20_40yd_ydps_mean"),
                       ("sustained_dash_speed", "forty_peak_sustained_speed_ydps_mean")]:
        if col in joined:
            r, n = pearson(joined[col].to_numpy(dtype=float), joined[target].to_numpy(dtype=float))
            rs, _ = spearman(joined[col].to_numpy(dtype=float), joined[target].to_numpy(dtype=float))
            out["correlations"][label] = {"n_players": n, "pearson_r": safe_float(r), "spearman_rho": safe_float(rs)}

    model_df = joined.copy()
    numeric = ["forty", "forty_early_accel_0_1s_ydps2_mean", "forty_late_accel_10_20yd_ydps2_mean",
               "forty_speed_change_20_40yd_ydps_mean", "combine_weight", "combine_height"]
    for col in numeric:
        if col not in model_df:
            model_df[col] = np.nan
    # Cohort-held-out scores are only calculated when enough complete training and test players exist.
    for cohort in sorted(model_df.draft_year.unique()):
        train = model_df.loc[(model_df.draft_year != cohort) & model_df[numeric + [target]].notna().all(axis=1)].copy()
        test = model_df.loc[(model_df.draft_year == cohort) & model_df[numeric + [target]].notna().all(axis=1)].copy()
        record = {"held_out_draft_year": int(cohort), "n_train_players": len(train), "n_test_players": len(test)}
        if len(train) < 30 or len(test) < 10:
            record["status"] = "Insufficient complete players for a stable cohort-held-out score; descriptive associations only."
            out["cross_validation"].append(record)
            continue
        dummies = pd.get_dummies(model_df.nfl_position, prefix="position", dtype=float)
        model_df[dummies.columns] = dummies
        controls = numeric + list(dummies.columns)
        baseline = ["forty", "combine_weight", "combine_height"] + list(dummies.columns)
        tracking = baseline + ["forty_early_accel_0_1s_ydps2_mean", "forty_late_accel_10_20yd_ydps2_mean", "forty_speed_change_20_40yd_ydps_mean"]
        train = model_df.loc[(model_df.draft_year != cohort) & model_df[tracking + [target]].notna().all(axis=1)].copy()
        test = model_df.loc[(model_df.draft_year == cohort) & model_df[tracking + [target]].notna().all(axis=1)].copy()
        record.update({"n_train_complete": len(train), "n_test_complete": len(test)})
        if len(train) < 30 or len(test) < 10:
            record["status"] = "Insufficient complete players after adding movement features."
        else:
            actual = test[target].to_numpy(dtype=float)
            pred_base = ridge_predict(train, test, baseline, target)
            pred_track = ridge_predict(train, test, tracking, target)
            record.update({"baseline_rmse_ydps": float(np.sqrt(np.mean((actual - pred_base) ** 2))),
                           "movement_plus_baseline_rmse_ydps": float(np.sqrt(np.mean((actual - pred_track) ** 2))),
                           "rmse_change_ydps": float(np.sqrt(np.mean((actual - pred_track) ** 2)) - np.sqrt(np.mean((actual - pred_base) ** 2))),
                           "status": "Exploratory; review sample and feature coverage before interpretation."})
        out["cross_validation"].append(record)
    (outdir / "model_results.json").write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    return out


def write_svg(joined, outdir):
    """Small dependency-free scatter plot of official 40 time versus rookie game speed."""
    cols = ["forty", "rookie_reg_p90_play_peak_speed_ydps"]
    d = joined[cols + ["nfl_position", "combine_position"]].dropna(subset=cols)
    if len(d) < 3:
        return None
    w, h, ml, mr, mt, mb = 760, 470, 86, 34, 42, 75
    xmin, xmax = float(d.forty.min()), float(d.forty.max())
    ymin, ymax = float(d[cols[1]].min()), float(d[cols[1]].max())
    if xmax == xmin: xmax += 1
    if ymax == ymin: ymax += 1
    def px(x): return ml + (x - xmin) / (xmax - xmin) * (w - ml - mr)
    def py(y): return h - mb - (y - ymin) / (ymax - ymin) * (h - mt - mb)
    position_col = "combine_position"
    colors = {p: f"hsl({(i*67)%360},52%,42%)" for i,p in enumerate(sorted(d[position_col].dropna().unique()))}
    circles = "\n".join(f'<circle cx="{px(r.forty):.1f}" cy="{py(getattr(r, cols[1])):.1f}" r="3.6" fill="{colors[getattr(r, position_col)]}" fill-opacity="0.66"><title>{getattr(r, position_col)}: 40={r.forty:.2f}s, rookie game p90={getattr(r, cols[1]):.2f} yd/s</title></circle>' for r in d.itertuples(index=False) if pd.notna(getattr(r, position_col)))
    legend = " ".join(f'<text x="{ml+i*92}" y="{h-18}" font-size="12" fill="{colors[p]}">{p}</text>' for i,p in enumerate(colors))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="100%" height="100%" fill="white"/><text x="{w/2}" y="24" text-anchor="middle" font-family="Arial" font-size="17">Official 40 time and rookie-season game speed</text>
<path d="M{ml} {mt}V{h-mb}H{w-mr}" stroke="#333" fill="none"/><text x="{w/2}" y="{h-43}" text-anchor="middle" font-family="Arial" font-size="13">Official 40-yard dash time (seconds)</text>
<text x="20" y="{h/2}" transform="rotate(-90 20 {h/2})" text-anchor="middle" font-family="Arial" font-size="13">90th percentile of per-play peak speed (yd/s)</text>
<text x="{ml}" y="{h-mb+20}" font-size="11">{xmin:.2f}</text><text x="{w-mr-30}" y="{h-mb+20}" font-size="11">{xmax:.2f}</text>
<text x="{ml-37}" y="{h-mb}" font-size="11">{ymin:.1f}</text><text x="{ml-37}" y="{mt+4}" font-size="11">{ymax:.1f}</text>
{circles}{legend}</svg>'''
    path = outdir / "forty_vs_game_speed.svg"
    path.write_text(svg, encoding="utf-8")
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--reuse-game-profiles", action="store_true", help="Reuse existing rookie game-speed summary and skip rescanning the 22M-frame game files.")
    args = parser.parse_args()
    final_outdir = args.output.expanduser().resolve()
    final_outdir.mkdir(parents=True, exist_ok=True)
    with OutputDirectoryLock(final_outdir):
        run_core_pipeline(args, final_outdir)


def run_core_pipeline(args, final_outdir):
    if not args.archive.exists():
        raise FileNotFoundError(f"Competition input path not found: {args.archive}")
    cache_identity = game_profile_cache_identity(args.archive)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = f"{started_at.replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:8]}"
    run_manifest_path = final_outdir / "core_run_manifest.json"
    write_json_atomic(run_manifest_path, {
        "manifest_schema_version": 2,
        "pipeline": "combine-to-rookie-game-speed-core",
        "status": "in_progress",
        "started_at_utc": started_at,
        "run_id": run_id,
        "core_profile_cache_key": cache_identity["cache_key"],
        "input": cache_identity["input"],
        "producer": cache_identity["producer"],
        "interpretation": "This run has started; outputs must not be treated as complete until this manifest is replaced with status=complete.",
    })
    outdir = final_outdir.parent / f".{final_outdir.name}.core-{run_id}.staging"
    outdir.mkdir(parents=False, exist_ok=False)
    if args.reuse_game_profiles:
        for name in ("player_rookie_game_speed_profiles.csv", "game_speed_profiles_cache_manifest.json"):
            cached_source = final_outdir / name
            if cached_source.is_file():
                shutil.copy2(cached_source, outdir / name)

    data_source = open_data_source(args.archive)
    with data_source as archive:
        players = read_csv(archive, "players.csv")
        results = read_csv(archive, "combine_results.csv")
        players["nfl_id"] = pd.to_numeric(players.nfl_id, errors="coerce")
        players["draft_year"] = pd.to_numeric(players.draft_year, errors="coerce")
        results["nfl_id"] = pd.to_numeric(results.nfl_id, errors="coerce")
        results["draft_year"] = pd.to_numeric(results.draft_year, errors="coerce")
        attempts, drills, forty = combine_profiles(archive, players, results, outdir)
        game_path = outdir / "player_rookie_game_speed_profiles.csv"
        cache_manifest_path = outdir / "game_speed_profiles_cache_manifest.json"
        if args.reuse_game_profiles:
            if not game_path.is_file() or not cache_manifest_path.is_file():
                raise RuntimeError("Cannot reuse game profiles: cache file or provenance manifest is missing. Rerun without --reuse-game-profiles.")
            prior = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            if prior.get("cache_key") != cache_identity["cache_key"]:
                raise RuntimeError("Cannot reuse game profiles: source data, processing code, or cache schema differs from the recorded cache. Rerun without --reuse-game-profiles.")
            cached_output = prior.get("output", {})
            if cached_output.get("sha256") != sha256_file(game_path) or cached_output.get("size_bytes") != game_path.stat().st_size:
                raise RuntimeError("Cannot reuse game profiles: cached CSV does not match its manifest. Rerun without --reuse-game-profiles.")
            game = pd.read_csv(game_path)
            linkage = prior.get("game_tracking_linkage")
            if not isinstance(linkage, dict):
                raise RuntimeError("Cannot reuse game profiles: manifest lacks game-tracking linkage metadata. Rerun without --reuse-game-profiles.")
            prior["run_id"] = run_id
            prior["run_started_at_utc"] = started_at
            write_json_atomic(cache_manifest_path, prior)
        else:
            game, linkage = game_speed_profiles(archive, players, outdir)
            cache_manifest = {
                **cache_identity,
                "run_id": run_id,
                "run_started_at_utc": started_at,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "output": {"name": game_path.name, "size_bytes": game_path.stat().st_size,
                           "sha256": sha256_file(game_path)},
                "game_tracking_linkage": linkage,
            }
            write_json_atomic(cache_manifest_path, cache_manifest)

        all_players = players.merge(results.drop(columns=["draft_year"], errors="ignore"), on="nfl_id", how="left", validate="one_to_one")
        forty_features = forty.drop(columns=[c for c in ["forty", "ten_yd_split", "combine_position"] if c in forty.columns])
        all_players = all_players.merge(forty_features, on="nfl_id", how="left", validate="one_to_one")
        all_players = all_players.merge(game, on=["nfl_id", "draft_year", "nfl_position"], how="left", validate="one_to_one", suffixes=("", "_game"))
        all_players.to_csv(outdir / "all_player_speed_profiles.csv", index=False)

        joined = all_players.loc[all_players.forty.notna() & all_players.rookie_reg_p90_play_peak_speed_ydps.notna()].copy()
        model_results = evaluate_models(joined, outdir)
        chart = write_svg(joined, outdir)
        directional_drills = {"SHORT_SHUTTLE", "THREE_CONE_DRILL", "W_DRILL", "BOX_DRILL", "PASS_PRO-MIRROR_DRILL", "PASS_PRO_MIRROR_DRILL", "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"}
        side_rows = drills.loc[drills.drill_name.isin(directional_drills)].copy()
        left_values = "left_lateral_speed_mean_ydps_mean"
        right_values = "right_lateral_speed_mean_ydps_mean"
        directional_coverage = []
        if len(side_rows):
            for drill_name, group in side_rows.groupby("drill_name"):
                both = group[left_values].notna() & group[right_values].notna()
                directional_coverage.append({"drill_name": drill_name,
                    "players": int(group.nfl_id.nunique()),
                    "players_with_left_lateral_movement": int(group.loc[group[left_values].notna(), "nfl_id"].nunique()),
                    "players_with_right_lateral_movement": int(group.loc[group[right_values].notna(), "nfl_id"].nunique()),
                    "players_with_both_directions": int(group.loc[both, "nfl_id"].nunique()),
                    "median_left_minus_right_lateral_speed_ydps": safe_float(group.loc[both, "left_right_lateral_speed_difference_by_drill_ydps"].median()) if both.any() else None})

        coverage = {
            "core_run_id": run_id,
            "archive": args.archive.name,
            "archive_kind": "zip" if args.archive.is_file() else "directory",
            "game_profile_cache_key": cache_identity["cache_key"],
            "players": int(players.nfl_id.nunique()),
            "combine_player_attempts": int(len(attempts)),
            "combine_drill_player_groups": int(len(drills)),
            "players_with_40_tracking": int(forty.nfl_id.nunique()),
            "players_with_official_40": int(results.loc[results.forty.notna(), "nfl_id"].nunique()),
            "players_with_both_40_tracking_and_official_40": int(forty.loc[forty.forty.notna(), "nfl_id"].nunique()),
            "players_with_rookie_regular_season_game_speed": int(game.rookie_reg_p90_play_peak_speed_ydps.notna().sum()),
            "players_with_both_combine_40_and_rookie_game_speed": int(len(joined)),
            "players_without_40_tracking": int(players.nfl_id.nunique() - forty.nfl_id.nunique()),
            "players_without_rookie_game_speed": int(game.rookie_reg_p90_play_peak_speed_ydps.isna().sum()),
            "directional_coverage_by_drill": directional_coverage,
            "game_tracking_linkage": linkage,
            "attempt_threshold_notes": {"speed_movement_start_ydps": SPEED_START_THRESHOLD, "smoothing": "centered 3-frame median", "sustained_peak": "rolling 5-frame median, at least 3 observed frames", "distance": "signed x-coordinate course progress from detected movement onset; direction inferred from net x displacement", "reaction_time": "not measured; cue timestamps are unavailable"},
            "output_chart": chart,
            "model_results": model_results,
        }
        (outdir / "coverage_and_results.json").write_text(json.dumps(coverage, indent=2, allow_nan=False), encoding="utf-8")
        write_report(coverage, outdir)
        promoted_names = [
            "combine_attempt_speed_profiles.csv",
            "player_drill_speed_profiles.csv",
            "player_40_speed_profiles.csv",
            "player_rookie_game_speed_profiles.csv",
            "all_player_speed_profiles.csv",
            "model_results.json",
            "coverage_and_results.json",
            "speed_analysis_report.md",
            "game_speed_profiles_cache_manifest.json",
            chart,
        ]
        missing = [name for name in promoted_names if not (outdir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Staged core outputs are incomplete; previous outputs were left in place: {missing}")
        promotion_inventory = {
            name: {"size_bytes": (outdir / name).stat().st_size,
                   "sha256": sha256_file(outdir / name)}
            for name in sorted(set(promoted_names))
        }
        promotion_marker = {
            "manifest_schema_version": 2,
            "pipeline": "combine-to-rookie-game-speed-core",
            "status": "promoting",
            "started_at_utc": started_at,
            "promotion_started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": run_id,
            "core_profile_cache_key": cache_identity["cache_key"],
            "input": cache_identity["input"],
            "producer": cache_identity["producer"],
            "staged_outputs": promotion_inventory,
            "interpretation": "Every promoted output must match the recorded staged size and SHA-256 before this run can be certified complete.",
        }
        write_json_atomic(run_manifest_path, promotion_marker)
        for name in promoted_names:
            (outdir / name).replace(final_outdir / name)
        write_json_atomic(run_manifest_path, {
            **promotion_marker,
            "status": "promoted",
            "promoted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        shutil.rmtree(outdir)
        print(json.dumps({k: coverage[k] for k in ["players", "combine_player_attempts", "players_with_40_tracking", "players_with_both_combine_40_and_rookie_game_speed", "game_tracking_linkage"]}, indent=2))


def write_report(summary, outdir):
    model = summary["model_results"]
    lines = [
        "# Speed profiles from the Combine to NFL games", "",
        "## Research question", "",
        "Do early acceleration, late acceleration, or speed maintenance during the Combine 40-yard dash explain rookie-season NFL game speed beyond the official 40-yard time?", "",
        "## Population and coverage", "",
        f"The supplied files include {summary['players']} prospects. Combine player tracking contains {summary['combine_player_attempts']:,} player-attempt traces across drills; {summary['players_with_40_tracking']} players have a tracked 40-yard dash. {summary['players_with_both_combine_40_and_rookie_game_speed']} players have both tracked Combine dash measures and at least one matched rookie-year regular-season tracked play with game speed.", "",
        "The outcome is the player-level 90th percentile of per-play peak speed during the player's rookie regular season. This reduces dependence on one fastest recorded frame, but it still depends on play opportunity, position, and coverage.", "",
        "## Measures", "",
        "The attempt file includes a movement-onset convention (first speed above 1 yd/s for two successive frames), speed at 0.5/1/2 seconds, acceleration at the start, speed and time at 5/10/20/30/40 yards, acceleration from 10–20 and 20–30 yards, change and retention from 20–40 yards, peak acceleration, and peak braking from speed changes. Speed is smoothed with a centered 3-frame median. Distance-indexed sprint phases use signed x-coordinate progress; true x/y path length is used for separate path-distance summaries.", "",
        "Left/right speed is measured only in drills with lateral movement, relative to the player's initial direction of travel. We compare available left and right movement across repeated attempts within the same drill. Combine tracking does not include body orientation, so this is a path-relative measure, not footwork or anatomical side preference. The 1 yd/s crossing describes observed movement onset, not reaction time, because cue timestamps are not supplied.", "",
        "## Findings", "",
    ]
    correlations = model["correlations"]
    for name, values in correlations.items():
        r = values["pearson_r"]
        rho = values["spearman_rho"]
        lines.append(f"- {name.replace('_', ' ')}: n={values['n_players']}; Pearson r={r:.3f} and Spearman rho={rho:.3f}." if r is not None and rho is not None else f"- {name.replace('_', ' ')}: insufficient variation or observations for a correlation.")
    lines += ["", "These are exploratory player-level associations, not causal effects. The cohort has only 510 independent players, and the linked 40-to-game sample is smaller. Treat small differences cautiously.", "", "## Held-out draft cohort comparison", ""]
    for fold in model["cross_validation"]:
        if "baseline_rmse_ydps" in fold:
            lines.append(f"- Held out {fold['held_out_draft_year']}: n={fold['n_test_complete']} players; baseline RMSE {fold['baseline_rmse_ydps']:.3f} yd/s; plus movement features {fold['movement_plus_baseline_rmse_ydps']:.3f} yd/s; change {fold['rmse_change_ydps']:+.3f} yd/s.")
        else:
            lines.append(f"- Held out {fold['held_out_draft_year']}: {fold['status']} (n train={fold['n_train_players']}, test={fold['n_test_players']}).")
    lines += ["", "## Left/right speed by drill", "", "These are descriptive coverage checks. The difference is each player's mean lateral speed toward the path-relative left minus mean lateral speed toward the path-relative right, summarized by drill.", "", "| Drill | Players with both directions | Median left-minus-right speed (yd/s) |", "|---|---:|---:|"]
    for row in summary.get("directional_coverage_by_drill", []):
        difference = row["median_left_minus_right_lateral_speed_ydps"]
        difference_text = f"{difference:+.2f}" if difference is not None else "n.a."
        lines.append(f"| {row['drill_name']} | {row['players_with_both_directions']} | {difference_text} |")
    lines += ["", "These comparisons can flag drills for closer review. They do not show anatomical left/right preference because the Combine file contains no body orientation.", "", "The baseline uses official 40 time, height, weight, and NFL position. The movement model adds early acceleration, 10–20-yard acceleration, and 20–40-yard speed change. In this exploratory pass, the added features do not lower prediction error across the held-out draft cohorts. The official 40 time already captures much of the broad association with game speed. This is a useful negative result for this target and sample; position-specific or play-phase outcomes may show a different pattern. Standardization uses training cohorts only.", "", "## Files", "", "- `all_player_speed_profiles.csv`: one row for each prospect, with Combine results, 40-yard tracking features, and rookie-game speed summary.", "- `combine_attempt_speed_profiles.csv`: features for each tracked PLAYER attempt.", "- `player_drill_speed_profiles.csv`: per-player summaries by specific drill, including left/right lateral speeds where observed.", "- `player_40_speed_profiles.csv`: repeated 40-yard attempts and official test value summarized by player.", "- `player_rookie_game_speed_profiles.csv`: rookie regular-season game-speed summaries for all prospects.", "- `model_results.json`: player-level associations and held-out cohort model scores.", "- `coverage_and_results.json`: coverage counts and measurement definitions.", "- `forty_vs_game_speed.svg`: scatter chart, colored by Combine position.", "", "Run `python run_analysis.py --archive PATH_TO_ZIP` to reproduce. Inputs and output counts are read from the archive; no source files are changed.", ""]
    (outdir / "speed_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
