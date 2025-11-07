from loguru import logger

class TrainingPipeline:
    def __init__(self, config):
        self.model_ops = config.model_ops
    def execute(self):
        logger.info("Starting Training Pipeline...")
        logger.info("Training Pipeline Completed...")