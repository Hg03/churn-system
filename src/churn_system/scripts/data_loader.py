from churn_system.scripts.data_validator import validate_data
import polars as pl

def load_raw(url: str) -> pl.DataFrame:
    df = pl.read_csv(url)
    return validate_data(df)