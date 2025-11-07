from churn_system.pipelines.inference_pipeline import InferencePipeline
from churn_system.utils.infer_utils import RequestData
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
from typing import Any, Optional

app = FastAPI(
    title="Full Stack API", description="A simple FastAPI application", version="1.0.0"
)


class InferenceRequest(BaseModel):
    input_data: RequestData = Field(..., description="Request Input Data")


class InferenceResponse(BaseModel):
    result: Any
    status: str
    message: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Hello, Stack!"}


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    input_data = request.input_data

    try:
        result = InferencePipeline(input_data=input_data).execute()
        return InferenceResponse(
            result=result, status="success", message="Inference completed successfully"
        )
    except Exception as e:
        return InferenceResponse(result={}, status="failed", message=f"Error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)