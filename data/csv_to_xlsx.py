import sys
import pandas as pd
from pathlib import Path

EXCEL_MAX_ROWS = 1_048_576


def csv_to_xlsx(csv_path: Path, xlsx_path: Path, sheet_name: str = "data"):
    df = pd.read_csv(csv_path, low_memory=False)
    sheet_name = sheet_name[:31]  # Excel sheet-name length limit
    if len(df) > EXCEL_MAX_ROWS - 1:
        raise ValueError(f"{csv_path} has {len(df)} rows, exceeds Excel's row limit")
    df.to_excel(xlsx_path, sheet_name=sheet_name, index=False, engine="openpyxl")
    print(f"{csv_path} -> {xlsx_path} ({df.shape[0]} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    csv_path = Path(sys.argv[1])
    xlsx_path = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.with_suffix(".xlsx")
    sheet_name = sys.argv[3] if len(sys.argv) > 3 else "data"
    csv_to_xlsx(csv_path, xlsx_path, sheet_name)
