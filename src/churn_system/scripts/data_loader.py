import polars as pl

def load_raw(url: str) -> pl.DataFrame:
    df = pl.read_csv(url)
    return df