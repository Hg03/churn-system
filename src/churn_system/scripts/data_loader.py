from churn_system.scripts.data_validator import validate_data
import polars as pl

def load_raw(url: str, save_path: str) -> pl.DataFrame:
    df = pl.read_csv(url)
    validate_data(df)
    df.write_parquet(save_path)
    return df