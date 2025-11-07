from churn_system.scripts.data_loader import load_raw
from churn_system.scripts.data_processor import preprocess, load_to_hopsworks
from churn_system.utils.feature_store_utils import get_fs
from loguru import logger

class FeaturePipeline:
    def __init__(self, config):
        self.paths = config.paths
        self.data_ops = config.data_ops
        self.pipeline = config.pipeline
    
    def yield_fs(self):
        return None if self.pipeline.type == 'local' else get_fs()

    def execute(self):
        logger.info("Starting Feature Pipeline...")
        self.raw_data = load_raw(url=self.data_ops.data_url, save_path=self.paths.raw_data)
        logger.info("Raw Data loaded and validated.")
        self.train_transformed, self.test_transformed = preprocess(df=self.raw_data, features=self.data_ops.features, strategies=self.data_ops.strategies, save_path=[self.paths.train_processed_data, self.paths.test_processed_data, self.paths.preprocessor])
        logger.info("Raw Data preprocessed.")
        load_to_hopsworks(fs=self.yield_fs(), training_data=self.train_transformed, testing_data=self.test_transformed)
        logger.info("Feature Pipeline Completed...")