"""
@author: mareiss
Version: 1.0
"""

import pandas as pd
import numpy as np

def align_and_interpolate(df_to_align, df_reference, method='linear', max_gap='48H', return_mask=False):
    """
    Align a DataFrame to match the reference DataFrame's dates using the specified interpolation method.

    """

    # Ensure datetime index
    if not isinstance(df_to_align.index, pd.DatetimeIndex):
        if 'Time' not in df_to_align.columns:
            raise ValueError("df_to_align must have a DatetimeIndex or a 'Time' column.")
        df_to_align = df_to_align.set_index('Time')

    if not isinstance(df_reference.index, pd.DatetimeIndex):
        if 'Time' not in df_reference.columns:
            raise ValueError("df_reference must have a DatetimeIndex or a 'Time' column.")
        df_reference = df_reference.set_index('Time')

    # Normalize max_gap
    max_gap = pd.Timedelta(max_gap)
    non_numeric_cols = df_to_align.select_dtypes(exclude=[np.number]).columns
    numeric_cols = df_to_align.select_dtypes(include=[np.number]).columns
    full_index = df_reference.index.union(df_to_align.index).sort_values()
    reindexed_numeric = df_to_align[numeric_cols].reindex(full_index)

    # Interpolate
    try:
        aligned_numeric_full = reindexed_numeric.interpolate(method=method)
    except ValueError:
        raise ValueError(f"Unsupported interpolation method: {method}")

    original_isna = reindexed_numeric.isna()

    blocked = pd.DataFrame(False, index=full_index, columns=numeric_cols)

    for col in numeric_cols:
        na = original_isna[col].to_numpy()
        idx = full_index.to_numpy()

        i = 0
        n = len(na)
        while i < n:
            if not na[i]:
                i += 1
                continue
            start = i
            while i < n and na[i]:
                i += 1
            end = i - 1

            run_start = pd.Timestamp(idx[start])
            run_end = pd.Timestamp(idx[end])
            run_duration = run_end - run_start

            if run_duration > max_gap:
                blocked.iloc[start:end + 1, blocked.columns.get_loc(col)] = True

    aligned_numeric_full = aligned_numeric_full.mask(blocked)

    aligned_numeric = aligned_numeric_full.reindex(df_reference.index)

    reindexed_non_numeric = df_to_align[non_numeric_cols].reindex(full_index) if len(non_numeric_cols) else None
    if reindexed_non_numeric is not None:
        aligned_non_numeric = reindexed_non_numeric.ffill().bfill().reindex(df_reference.index)
        aligned_values = pd.concat([aligned_numeric, aligned_non_numeric], axis=1)
    else:
        aligned_values = aligned_numeric.copy()

    aligned_values.index = df_reference.index

    if len(numeric_cols):
        valid_mask = aligned_numeric.notna().all(axis=1)
    else:
        valid_mask = pd.Series(True, index=df_reference.index)

    out = aligned_values.reset_index()

    if return_mask:
        return out, valid_mask.reset_index(drop=True)
    return out


def apply_time_rolling_average(df, value_col, time_col="Time", rolling_days=0, min_periods=1):
    """
    Applies a time-based rolling mean to a selected column in a time series DataFrame.

            Parameters:
                df (pandas.DataFrame):
                    Input data. Must contain:
                        - time_col (datetime-like): timestamps
                        - value_col (float): values to smooth

                value_col (str):
                    Name of the column to smooth (for example, "Model" or "Obs").

                time_col (str, optional):
                    Name of the time column.
                    Default is "Time".

                rolling_days (int, optional):
                    Width of the rolling window in days.
                    A value of 0 returns the original series without smoothing.
                    Values are constrained to the range [0, 5] to match the UI slider.
                    Default is 0.

                min_periods (int, optional):
                    Minimum number of samples required within each rolling window
                    to compute the mean.
                    Passed directly to pandas rolling(...).mean().
                    Default is 1.

            Returns:
                pandas.DataFrame:
                    DataFrame containing:
                        - time_col: timestamps
                        - value_col: smoothed values
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[time_col, value_col])

    if time_col not in df.columns:
        raise ValueError(f"apply_time_rolling_average: '{time_col}' column not found.")
    if value_col not in df.columns:
        raise ValueError(f"apply_time_rolling_average: '{value_col}' column not found.")

    rolling_days = int(rolling_days) if rolling_days is not None else 0
    rolling_days = max(0, min(5, rolling_days))

    tmp = df[[time_col, value_col]].copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col])

    # Sort by time before applying time-based rolling
    tmp = tmp.sort_values(time_col)

    if rolling_days == 0:
        return tmp.reset_index(drop=True)

    rolled = (
        tmp.set_index(time_col)[value_col]
        .rolling(f"{rolling_days}D", min_periods=min_periods)
        .mean()
        .reset_index()
    )

    rolled.columns = [time_col, value_col]
    return rolled.reset_index(drop=True)