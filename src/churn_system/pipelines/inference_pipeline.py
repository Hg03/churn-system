from typing import Any


class InferencePipeline:
    def __init__(self, input_data):
        self.input_data = input_data

    def execute(self) -> Any:
        print("Take the features, take the model, serve the predictions..")
        return {}