from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from churn_system.utils.feature_store_utils import get_fg, add_feature_descriptions
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from omegaconf import DictConfig
from typing import Dict, List, Any
from loguru import logger
import polars as pl
import pickle

def build_imputer(features: Dict[str, List[str]], strategies: Dict[str, List[str]]):
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

def build_encoder(features: Dict[str, List[str]], strategies: Dict[str, List[str]]):
    nominal_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    return ColumnTransformer(
        [("nominal encoder", nominal_encoder, list(features.categoric))],
        remainder='passthrough'
    ).set_output(transform='polars')

def build_preprocessor(df: pl.DataFrame, features: Dict[str, List[str]], strategies: Dict[str, List[str]]):
    imputer = build_imputer(features=features, strategies=strategies)
    encoder = build_encoder(features=features, strategies=strategies)
    return Pipeline([("imputer", imputer), ("postprocessor1", FunctionTransformer(postprocessor)), ("encoder", encoder), ('postprocessor2', FunctionTransformer(postprocessor))])

def split(X: pl.DataFrame, y: pl.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    return X_train, X_test, y_train, y_test

def save_transformed(transformed_data: List[pl.DataFrame], targets: List[pl.DataFrame], save_path: List[str]):
    train_transformed = transformed_data[0].with_row_count(name="id").with_columns(
        pl.Series("churned", targets[0].to_series())
    )
    test_transformed = transformed_data[1].with_row_count(name="id").with_columns(
        pl.Series("churned", targets[1].to_series())
    )
    train_transformed.write_parquet(save_path[0])
    test_transformed.write_parquet(save_path[1])
    return train_transformed, test_transformed

def preprocess(df: pl.DataFrame, features: Dict[str, List[str]], strategies: Dict[str, List[str]], save_path: List[str]):
    preprocessor = build_preprocessor(df=df, features=features, strategies=strategies)
    X,y = df.select(pl.col(features.X)), df.select(pl.col(features.y))
    X_train, X_test, y_train, y_test = split(X=X, y=y)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    train_transformed, test_transformed = save_transformed(transformed_data=[X_train_transformed, X_test_transformed], targets=[y_train, y_test], save_path=save_path)
    with open(save_path[2], 'wb') as picklefile:
        pickle.dump(preprocessor, picklefile)
    
    return [train_transformed, test_transformed]


def load_to_hopsworks(fs: Any, config: DictConfig, training_data: pl.DataFrame, testing_data: pl.DataFrame):
    if fs:
        train_fg, test_fg = get_fg(fs, config=config), get_fg(fs, config=config)
        train_fg.insert(training_data.to_pandas())
        test_fg.insert(testing_data.to_pandas())
        add_feature_descriptions(train_fg, config)
        add_feature_descriptions(test_fg, config)
        
    else:
        logger.info('Local Pipeline type enabled, so no feature store.')