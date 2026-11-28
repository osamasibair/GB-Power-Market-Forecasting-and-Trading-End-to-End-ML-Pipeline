"""Copy the tables the model runs need (features, transformer_forecasts, transformer_calibration) from this machine's
database (DATABASE_URL, or the local default) to the shared research database every RunPod machine uses
(SHARED_DATABASE_URL), then read each back and check it matches exactly. Run once on the Mac before the runs:
    SHARED_DATABASE_URL=... python src/copy_to_shared.py
After the runs, --back copies the forecast tables (with every model's final forecasts added) back to this machine's
database, and writes the files the runs saved (each model's lookback and tuned settings) into models/:
    SHARED_DATABASE_URL=... python src/copy_to_shared.py --back"""
import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from db import engine as local_engine

tables = ["features", "transformer_forecasts", "transformer_calibration"]
results = ["transformer_forecasts", "transformer_calibration"]


def copy(name, source, target):
    """Copy one table and check the copy matches the original exactly."""
    table = pd.read_sql_table(name, source, index_col="timestamp")
    table.to_sql(name, target, if_exists="replace", index_label="timestamp", chunksize=10_000)
    copied = pd.read_sql_table(name, target, index_col="timestamp")
    pd.testing.assert_frame_equal(table, copied, check_exact=True)
    print(f"copied {name}: {len(table):,} rows, {table.shape[1]} columns, identical")


def bring_back_files():
    """Write every file the runs kept on the shared board (parallel.keep_file) to the same path here."""
    from parallel import Board

    for name, text in Board(None, "files").files().items():
        path = Path(name)
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--back", action="store_true", help="copy the results back to this machine")
    args = parser.parse_args()
    url = os.environ.get("SHARED_DATABASE_URL")
    if not url:
        raise SystemExit("set SHARED_DATABASE_URL to the shared research database first")
    shared = create_engine(url, pool_pre_ping=True)
    if args.back:
        for name in results:
            copy(name, shared, local_engine)
        os.environ["OPTUNA_STORAGE"] = url  # the board is in the shared database
        bring_back_files()
    else:
        for name in tables:
            copy(name, local_engine, shared)
