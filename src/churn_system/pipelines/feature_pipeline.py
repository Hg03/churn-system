from churn_system.scripts.data_loader import load_raw
from churn_system.scripts.data_processor import preprocess
from loguru import logger

class FeaturePipeline:
    def __init__(self, config):
        self.paths = config.paths
        self.data_ops = config.data_ops
        self.pipeline = config.pipeline
    def execute(self):
        logger.info("Starting Feature Pipeline...")
        self.raw_data = load_raw(url=self.data_ops.data_url, save_path=self.paths.raw_data)
        logger.info("Raw Data loaded and validated.")
        self.transformed_data = preprocess(df=self.raw_data, features=self.data_ops.features, strategies=self.data_ops.strategies, save_path=[self.paths.train_processed_data, self.paths.test_processed_data])
        logger.info("Raw Data preprocessed.")
        
        logger.info("Feature Pipeline Completed...")