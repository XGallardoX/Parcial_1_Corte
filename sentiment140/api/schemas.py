from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class PredictInput(BaseModel):
    text:  Optional[str]       = Field(None, example="I love this product!")
    texts: Optional[List[str]] = Field(None, example=["Amazing!", "Terrible."])
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "I absolutely love this!"},
                {"texts": ["Amazing!", "Worst experience ever."]}
            ]
        }
    }

class PredictionItem(BaseModel):
    text: str; prediction: str; label: int
    confidence: float; probabilities: Dict[str, float]

class SinglePredictResponse(PredictionItem):
    inference_time_sec: float

class BatchPredictResponse(BaseModel):
    predictions: List[PredictionItem]
    count: int; inference_time_sec: float
