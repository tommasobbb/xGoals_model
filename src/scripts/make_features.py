from pathlib import Path
import traceback
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import (
    SHOTS_OUT_PATH,
    FEATURES_OUT_PATH,
)

# --- Constants for StatsBomb pitch geometry ---
GOAL_X = 120.0
GOAL_Y = 40.0
LEFT_POST_Y = 36.0
RIGHT_POST_Y = 44.0

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def distance_and_angle(x: float, y: float) -> tuple[float, float]:
    """
    Calculate the distance to goal and shot angle for a given shot position.
    
    For StatsBomb data, the goal is always at x=120, y=40 (center of goal).
    The goal spans from y=36 to y=44 (8 yards wide).
    
    Parameters
    ----------
    x : float
        The x-coordinate of the shot position
    y : float  
        The y-coordinate of the shot position
        
    Returns
    -------
    tuple[float, float]
        A tuple containing (distance_to_goal, shot_angle) where:
        - distance_to_goal: Euclidean distance from shot position to goal center
        - shot_angle: Angle subtended by the goal as seen from the shot position (in degrees)
    """
    # Distance to goal center
    distance_to_goal = np.sqrt((GOAL_X - x)**2 + (GOAL_Y - y)**2)
    
    # Vectors from shot position to each goal post
    vec_to_left = np.array([GOAL_X - x, LEFT_POST_Y - y])
    vec_to_right = np.array([GOAL_X - x, RIGHT_POST_Y - y])
    
    # Calculate angle between the two vectors using dot product
    dot_product = np.dot(vec_to_left, vec_to_right)
    mag_left = np.linalg.norm(vec_to_left)
    mag_right = np.linalg.norm(vec_to_right)
    
    # Handle edge case where shot is exactly at a goal post
    if mag_left == 0 or mag_right == 0:
        shot_angle = 0.0
    else:
        # Clamp cosine value to [-1, 1] to handle numerical errors
        cos_angle = np.clip(dot_product / (mag_left * mag_right), -1.0, 1.0)
        shot_angle = np.arccos(cos_angle)
    
    return distance_to_goal, np.degrees(shot_angle)

def _in_shot_cone(px: float, py: float, sx: float, sy: float) -> bool:
    """
    Check whether a point lies inside the shot cone.

    The shot cone is defined at the shooter position ``(sx, sy)`` by the
    two rays from the shooter to the left and right goalposts. This
    function returns whether the test point ``(px, py)`` falls inside
    that angular region.

    Parameters
    ----------
    px : float
        x-coordinate of the test point.
    py : float
        y-coordinate of the test point.
    sx : float
        x-coordinate of the shooter.
    sy : float
        y-coordinate of the shooter.

    Returns
    -------
    bool
        ``True`` if the point lies within the angular cone formed by the
        two goalposts as seen from the shooter, ``False`` otherwise.
    """
    # Vector from shooter to test point
    test_vec = np.array([px - sx, py - sy])
    
    # Vectors from shooter to goal posts
    left_post_vec = np.array([GOAL_X - sx, LEFT_POST_Y - sy])
    right_post_vec = np.array([GOAL_X - sx, RIGHT_POST_Y - sy])
    
    # Handle edge cases where shooter is at goal line or beyond
    if sx >= GOAL_X:
        return False
    
    # Handle case where test point is behind the shooter (negative x direction)
    if px <= sx:
        return False
    
    # Cross product of left_post_vec × test_vec
    cross_left = left_post_vec[0] * test_vec[1] - left_post_vec[1] * test_vec[0]
    
    # Cross product of test_vec × right_post_vec  
    cross_right = test_vec[0] * right_post_vec[1] - test_vec[1] * right_post_vec[0]
    
    # Point is inside the cone if:
    return cross_left >= 0 and cross_right >= 0

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def compute_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes geometry-based features for shots.

    Given a DataFrame with shot coordinates (`x`, `y`), this function
    computes the `distance_to_goal` and `shot_angle` for each shot.
    It applies the `distance_and_angle` helper function to each row.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing raw shot data. It must have
        'x' and 'y' columns.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with two new columns added:
        - `distance_to_goal`
        - `shot_angle`
    """
    xy = df[["x", "y"]].to_numpy(dtype=float, copy=False)

    geom = np.array([distance_and_angle(x, y) for x, y in xy], dtype=float)
    df["distance_to_goal"] = geom[:, 0]
    df["shot_angle"] = geom[:, 1]

    return df

def _extract_gk(freeze_frame: list | None) -> tuple[float, float] | None:
    """
    Extract the goalkeeper's (x, y) location from a freeze-frame.

    Parameters
    ----------
    freeze_frame : list of dict or None

    Returns
    -------
    tuple of float or None
        The (x, y) coordinates of the goalkeeper if found,
        otherwise ``None``.
    """
    # Properly handle None
    if freeze_frame is None:
        return None
    
    # Convert to list if it's a numpy array or other iterable
    try:
        if hasattr(freeze_frame, 'tolist'):
            freeze_frame = freeze_frame.tolist()
        elif not isinstance(freeze_frame, (list, tuple)):
            freeze_frame = list(freeze_frame)
    except (TypeError, AttributeError):
        return None
    
    for p in freeze_frame:
        if not p.get("teammate") and p.get("position", {}).get("name") == "Goalkeeper":
            loc = p.get("location")
            # Check if loc exists and has at least 2 elements
            if loc is not None and len(loc) >= 2:
                try:
                    # Convert to float, handling numpy arrays/pandas objects
                    x_val = loc[0]
                    y_val = loc[1]
                    
                    # Handle numpy/pandas objects
                    if hasattr(x_val, 'item'):
                        x_val = x_val.item()
                    if hasattr(y_val, 'item'):
                        y_val = y_val.item()
                        
                    return float(x_val), float(y_val)
                except (ValueError, TypeError, IndexError) as e:
                    print(f"Error converting goalkeeper location: {e}")
                    continue
    return None

def _split_attack_def(
    freeze_frame: list[dict] | None, 
    shooter_xy: tuple[float, float]
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Split freeze-frame players into defenders and teammates.

    Parameters
    ----------
    freeze_frame : list of dict or None
        List of player dictionaries from the freeze-frame..
    shooter_xy : tuple of float
        Coordinates (x, y) of the shooter.

    Returns
    -------
    defenders_xy : list of tuple of float
        List of (x, y) coordinates for defenders (opponents excluding the
        goalkeeper).
    teammates_xy : list of tuple of float
        List of (x, y) coordinates for teammates (excluding the shooter).
    """
    defenders, teammates = [], []
    sx, sy = shooter_xy

    # Properly handle None
    if freeze_frame is None:
        return [], []

    gk_xy = _extract_gk(freeze_frame)

    for p in freeze_frame:
        loc = p.get("location", None)
        if loc is not None and len(loc) >= 2:
            try:
                # Convert to float, handling numpy arrays/pandas objects
                x_val = loc[0]
                y_val = loc[1]
                
                # Handle numpy/pandas objects
                if hasattr(x_val, 'item'):
                    x_val = x_val.item()
                if hasattr(y_val, 'item'):
                    y_val = y_val.item()
                    
                x, y = float(x_val), float(y_val)
            except (ValueError, TypeError, IndexError) as e:
                print(f"Error converting location: {e}")
                continue
        else:
            continue
        # skip the shooter if present
        if abs(x - sx) < 1e-6 and abs(y - sy) < 1e-6:
            continue
        # skip the goalkeeper if present
        if gk_xy is not None and abs(x - gk_xy[0]) < 1e-6 and abs(y - gk_xy[1]) < 1e-6:
            continue
        if p.get("teammate", False):
            teammates.append((x, y))
        else:
            defenders.append((x, y))
    return defenders, teammates


def _in_penalty_area(x: float, y: float) -> bool:
    """
    Check whether a point lies inside the penalty area.

    Parameters
    ----------
    x : float
        x-coordinate of the point.
    y : float
        y-coordinate of the point.

    Returns
    -------
    bool
        ``True`` if the point lies inside the penalty area,
        ``False`` otherwise.
    """
    return (x >= 103.5) and (19.84 <= y <= 60.16)


def freeze_frame_features(
    freeze_frame: list[dict] | None,
    sx: float,
    sy: float,
) -> dict[str, int | float]:
    """
    Compute identity-agnostic freeze-frame features for a shot event.

    Given a freeze-frame (players' positions at the shot time) and the
    shooter's coordinates, this function extracts goalkeeper (GK) context,
    defensive pressure, and attacking support features.

    Parameters
    ----------
    freeze_frame : list of dict or None
    sx : float
        Shooter x-coordinate.
    sy : float
        Shooter y-coordinate.

    Returns
    -------
    dict of {str: int or float}
        A mapping with the freeze frame features.
    """
    # GK
    gk_xy = _extract_gk(freeze_frame)
    if gk_xy is not None:
        gx, gy = gk_xy
        gk_depth = max(0.0, GOAL_X - gx)
        gk_lateral_offset = abs(gy - GOAL_Y)
        gk_in_shot_cone = int(_in_shot_cone(gx, gy, sx, sy))
    else:
        gk_depth = np.nan
        gk_lateral_offset = np.nan
        gk_in_shot_cone = 0

    # Others
    defenders_xy, teammates_xy = _split_attack_def(freeze_frame, (sx, sy))
    
    # Defensive pressure metrics
    if defenders_xy:
        dists = [np.sqrt((sx - x)**2 + (sy - y)**2) for (x, y) in defenders_xy]
        closest_defender_distance = float(min(dists))
        defenders_within_1m = int(sum(d <= 1.0 for d in dists))
        defenders_within_2m = int(sum(d <= 2.0 for d in dists))
        defenders_within_3m = int(sum(d <= 3.0 for d in dists))
        defenders_within_5m = int(sum(d <= 5.0 for d in dists))
        defenders_in_shot_cone = int(
            sum(_in_shot_cone(x, y, sx, sy) for (x, y) in defenders_xy)
        )
    else:
        closest_defender_distance = np.nan
        defenders_within_1m = defenders_within_2m = defenders_within_3m = defenders_within_5m = 0
        defenders_in_shot_cone = 0

    # Attacking support
    teammates_in_box = int(sum(_in_penalty_area(x, y) for (x, y) in teammates_xy))
    teammates_closer_than_shooter = int(sum(x > sx for (x, _y) in teammates_xy))

    return {
        "gk_depth": gk_depth,
        "gk_lateral_offset": gk_lateral_offset,
        "gk_in_shot_cone": gk_in_shot_cone,
        "closest_defender_distance": closest_defender_distance,
        "defenders_within_1m": defenders_within_1m,
        "defenders_within_2m": defenders_within_2m,
        "defenders_within_3m": defenders_within_3m,
        "defenders_within_5m": defenders_within_5m,
        "defenders_in_shot_cone": defenders_in_shot_cone,
        "teammates_in_box": teammates_in_box,
        "teammates_closer_than_shooter": teammates_closer_than_shooter,
    }


def add_freeze_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute freeze-frame features for each row in the input DataFrame.

    This function processes a DataFrame of shot events, where each row may 
    include a `freeze_frame` describing player and goalkeeper positions 
    at the time of the shot. It extracts spatial and tactical features 
    related to defenders, teammates, and the goalkeeper. If no 
    `freeze_frame` column exists, default feature values are added.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with additional freeze-frame feature columns.
    """
    # Define defaults once at function level
    feature_defaults = {
        "gk_depth": np.nan,
        "gk_lateral_offset": np.nan,
        "closest_defender_distance": np.nan,
        "gk_in_shot_cone": 0,
        "defenders_within_1m": 0,
        "defenders_within_2m": 0,
        "defenders_within_3m": 0,
        "defenders_within_5m": 0,
        "defenders_in_shot_cone": 0,
        "teammates_in_box": 0,
        "teammates_closer_than_shooter": 0,
    }
    
    if "freeze_frame" not in df.columns:
        for col, default in feature_defaults.items():
            df[col] = default
        return df

    def _row_feats(r):
        ff = r.get("freeze_frame", [])
        sx, sy = float(r["x"]), float(r["y"])
        
        try:
            return freeze_frame_features(ff, sx, sy)
        except Exception as e:
            print(e)
            traceback.print_exc()
            return feature_defaults
    
    # Apply function and get new features
    ff_df = df.apply(_row_feats, axis=1, result_type="expand")
    
    return pd.concat([df, ff_df], axis=1)

def check_booleans(df: pd.DataFrame, boolean_cols: Iterable[str]) -> pd.DataFrame:
    """
    Ensures specified columns are of boolean type. Missing
    values (`NaN`) are replaced with `False`.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    boolean_cols : Iterable[str]
        A collection of column names to convert to boolean.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the specified columns cast to boolean type.
        If a column in `boolean_cols` does not exist in the DataFrame,
        it is ignored.
    """
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype('boolean').fillna(False)
    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: Iterable[str],
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Performs one-hot encoding on specified categorical columns.

    This function prepares and one-hot encodes a list of categorical
    columns. Missing values are filled with "Unknown" before encoding
    to ensure all categories are represented.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the categorical columns.
    categorical_cols : Iterable[str]
        A collection of column names to be one-hot encoded.
    drop_original : bool, default=True
        If `True`, the original categorical columns are dropped
        from the output DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with new, one-hot encoded columns. The new
        columns are prefixed with their original name and a separator (`=`).
    """
    for col in categorical_cols:
        if col not in df.columns:
            continue
        # Normalize type and missing values
        df[col] = df[col].astype("string").fillna("Unknown")

    dummies = pd.get_dummies(
        df[list(categorical_cols)],
        prefix=categorical_cols,
        prefix_sep="=",
        dtype=bool,
        drop_first=True,
    )

    df_out = pd.concat([df, dummies], axis=1)
    if drop_original:
        df_out = df_out.drop(columns=list(categorical_cols), errors="ignore")
    return df_out


def select_and_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and orders the final feature set for a model.

    This function builds the final DataFrame to be used for model training
    by selecting the appropriate features, dropping raw intermediate data,
    and ordering the columns for a clean and consistent output.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing engineered features and
        one-hot encoded categorical columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the final, cleaned feature set. Columns include:
        `match_id`, engineered numeric features, boolean features,
        one-hot encoded categorical features, and the `is_goal` label.
    """
    # Identify boolean and engineered columns
    boolean_cols = ["shot_first_time", "under_pressure"]
    engineered_numeric = ["distance_to_goal", "shot_angle"]

    # One-hot columns (all uint8 dummies)
    dummy_cols = [c for c in df.columns if "=" in c]

    # New: Add freeze-frame features to a separate list
    freeze_frame_cols = [
        "gk_depth", "gk_lateral_offset", "gk_in_shot_cone",
        "closest_defender_distance", "defenders_within_1m",
        "defenders_within_2m", "defenders_within_3m",
        "defenders_within_5m", "defenders_in_shot_cone",
        "teammates_in_box", "teammates_closer_than_shooter"
    ]

    base_cols = []
    if "match_id" in df.columns:
        base_cols.append("match_id")
    if "id" in df.columns:
        base_cols.append("id")   

    # Label
    label = ["is_goal"] if "is_goal" in df.columns else []
    xg_goals = ["shot_statsbomb_xg"]

    ordered = base_cols + engineered_numeric + boolean_cols + dummy_cols + freeze_frame_cols + label + xg_goals

    # Filter to existing columns only
    existing = [c for c in ordered if c in df.columns]

    # Drop raw coordinates and intermediate columns
    drop_cols = ["x", "y", "play_pattern", "shot_type", "shot_body_part", "shot_technique"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Reorder
    return df[existing]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw shot data into a feature DataFrame for modeling.

    This function takes a DataFrame of raw shot events and engineers
    new features, cleans existing ones, and prepares the data for
    training a machine learning model.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing raw shot events, with columns
        like `x`, `y`, `play_pattern`, `shot_body_part`, etc.

    Returns
    -------
    pd.DataFrame
        A feature-engineered DataFrame with one row per shot.
    """
    # Geometry
    df = compute_geometry_features(df)

    # Freeze-frame derived features
    df = add_freeze_frame_features(df)

    # Booleans
    df = check_booleans(df, boolean_cols=["shot_first_time", "under_pressure"])

    # Categoricals -> one-hot
    df = encode_categoricals(
        df,
        categorical_cols=["play_pattern", "shot_type", "shot_body_part", "shot_technique"],
        drop_original=True,
    )

    # Final selection/order
    df = select_and_order_columns(df)

    # Remove rows with missing engineered geometry
    df = df.dropna(subset=["distance_to_goal", "shot_angle"]).reset_index(drop=True)

    return df


def main():
    """
    Reads a shots dataset, engineers features, and saves the result.

    This script orchestrates the feature engineering pipeline. It loads
    the raw shot data, performs transformations to create new features,
    and then saves the final dataset to a Parquet file for model training.
    It also prints key statistics about the data, such as class balance.
    """
    # 1) Reads shots parquet
    print(f"Reading shots from: {SHOTS_OUT_PATH}")
    df = pd.read_parquet(SHOTS_OUT_PATH)

    # Print original columns
    print("\nColumns BEFORE feature engineering:")
    print(df.columns.tolist())

    # 2) Build features
    features_df = build_features(df)
    print(f"\nShots available: {len(df)}")

    # Print new columns
    print("\nColumns AFTER feature engineering:")
    print(features_df.columns.tolist())

    # 3) Check class balance (goals vs. non-goals)
    goal_count = features_df["is_goal"].sum()
    non_goal_count = len(features_df) - goal_count

    print(f"\nNumber of goals: {goal_count}")
    print(f"Number of non-goals: {non_goal_count}")

    # Ensure output directory exists
    out_path = Path(FEATURES_OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 4) Print statistics and check for NaNs
    print("\n--- Feature Dataset Statistics ---")
    print(features_df.describe().T)
    print("\n--- Missing values (NaNs) per column ---")
    print(features_df.isnull().sum())
    print(f"\nTotal rows with at least one NaN: {features_df.isnull().any(axis=1).sum()}")

    # 5) Drop rows with NaNs
    before_drop = len(features_df)
    features_df = features_df.dropna()
    after_drop = len(features_df)
    print(f"\nDropped {before_drop - after_drop} rows with NaNs. Remaining rows: {after_drop}")

    # 6) Save as parquet
    features_df.to_parquet(out_path, index=False)
    print(f"\nFeatures saved to: {out_path}")
    print("\nPreview:")
    print(features_df.head())

if __name__ == "__main__":
    main()
