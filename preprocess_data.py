"""
Preprocess long-format CSV (date, data, cols) into wide-format for ExDM.
Mimics GCGNet's read_data() pivoting logic: each unique 'cols' value becomes
a column, with the 'OT' column placed last as the target/endogenous variable.
All other columns serve as exogenous features.

Usage: python preprocess_data.py <input.csv> <output.csv>
"""
import sys
import pandas as pd


def preprocess(input_path: str, output_path: str) -> None:
    raw = pd.read_csv(input_path)

    # Determine number of points per group (GCGNet-style pivot)
    n_points = raw["cols"].value_counts().max()
    n_cols = len(raw) // n_points
    cols_name = raw["cols"].unique()

    # Build wide-format DataFrame: date + one column per unique cols value
    df = pd.DataFrame()
    df["date"] = raw.iloc[:n_points, 0]
    for j in range(n_cols):
        df[cols_name[j]] = raw.iloc[j * n_points:(j + 1) * n_points, 1].values

    df["date"] = pd.to_datetime(df["date"])

    # Reorder so OT (target) is the last data column
    data_cols = [c for c in df.columns if c != "date"]
    if "OT" in data_cols:
        data_cols.remove("OT")
        data_cols.append("OT")
    df = df[["date"] + data_cols]

    df.to_csv(output_path, index=False)
    print(f"Preprocessed: {len(data_cols)} columns ({data_cols[:-1]} -> exog, "
          f"{data_cols[-1:]} -> endo), {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    preprocess(sys.argv[1], sys.argv[2])
