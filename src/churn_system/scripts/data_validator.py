from churn_system.datamodels.raw_data_model import RawDataValidator
from pydantic import ValidationError
import polars as pl

def validate_data(df: pl.DataFrame):
    valid_rows = []
    invalid_rows = []
    if len(df) == 0:
        raise ValueError("No Data Found")
    idx = 0
    for row in df.iter_rows(named=True):
        try:
            record = RawDataValidator(**row)
            valid_rows.append(record.model_dump())
            idx += 1
        except ValidationError as e:
            invalid_rows.append({"row_index": idx, "errors": e.errors()})