full:
    @echo "Running Full Pipeline (Feature + Training)"
    @python app/training_endpoint.py

feature:
    @echo "Running Feature Pipeline"
    @python app/training_endpoint.py pipeline.stage=feature

training:
    @echo "Running Training Pipeline"
    @python app/training_endpoint.py pipeline.stage=training

inference:
    @echo "Running Inference Pipeline"
    @fastapi dev app/inference_api.py