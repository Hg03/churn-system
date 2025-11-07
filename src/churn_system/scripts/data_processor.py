from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Dict, List
import polars as pl

def build_imputer(df: pl.DataFrame, features: Dict[str, List[str]], strategies: Dict[str, List[str]]):
    numeric_imputer = SimpleImputer(strategy=strategies.numeric)
    categoric_imputer = SimpleImputer(strategy=strategies.categoric)
    return ColumnTransformer(
        [("numeric imputer", numeric_imputer, list(features.numeric)),
        ("categoric imputer", categoric_imputer, list(features.categoric))],
        remainder='passthrough'
    ).set_output(transform='polars')


def postprocessor(df):
    df.columns = [col[col.rfind('__')+2:] for col in df.columns]
    return df

def build_encoder(df: pl.DataFrame, features: Dict[str, List[str]], strategies: Dict[str, List[str]]):
    nominal_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    encoder = ColumnTransformer(
        [("nominal encoder", nominal_encoder, list(features.categoric))],
        remainder='passthrough'
    ).set_output(transform='polars')

def build_preprocessor(df: pl.DataFrame, features: Dict[str, List[str]], strategies: Dict[str, List[str]]):
    imputer = build_imputer(df=df, features=features, strategies=strategies)
    encoder = build_encoder(df, features=features, strategies=strategies)
    return Pipeline([("imputer", imputer), ("postprocessor", FunctionTransformer(postprocessor)), ("encoder", encoder)])

def split(X: pl.DataFrame, y: pl.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    return X_train, X_test, y_train, y_test

def save_transformed(transformed_data: List[pl.DataFrame], targets: List[pl.DataFrame], save_path: List[str]):
    train_transformed = transformed_data[0].with_columns(
        pl.Series("churned", targets[0].to_series())
    )
    test_transformed = transformed_data[1].with_columns(
        pl.Series("churned", targets[1].to_series())
    )
    train_transformed.write_parquet(save_path[0])
    test_transformed.write_parquet(save_path[1])

def preprocess(df: pl.DataFrame, features: Dict[str, List[str]], strategies: Dict[str, List[str]], save_path: List[str]):
    preprocessor = build_preprocessor(df=df, features=features, strategies=strategies)
    X,y = df.select(pl.col(features.X)), df.select(pl.col(features.y))
    X_train, X_test, y_train, y_test = split(X=X, y=y)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    save_transformed(transformed_data=[X_train_transformed, X_test_transformed], targets=[y_train, y_test], save_path=save_path)