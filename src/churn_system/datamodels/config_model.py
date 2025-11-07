from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class paths:
    raw: str
    processed: str
    raw_data: str
    processed_data: str
    preprocessor: str
    model: str

@dataclass
class data_ops:
    url: str

@dataclass
class model_ops:
    default_model: str

@dataclass
class config_model:
    paths: paths
    data_ops: data_ops
    model_ops: model_ops