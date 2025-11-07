from churn_system.scripts.data_loader import load_raw
from loguru import logger

class FeaturePipeline:
    def __init__(self, config):
        self.data_ops = config.data_ops
    def execute(self):
        logger.info("Starting Feature Pipeline...")
        self.raw_data = load_raw(url=self.data_ops.data_url)
        logger.info("Feature Pipeline Completed...")