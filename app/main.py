import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

class FileURLRequest(BaseModel):
    file_url: str

@app.post("/predict")
async def predict(request: FileURLRequest):
    try:
        file_url = request.file_url
        logging.debug(f"🔍 Received file URL: {file_url}")

        # 🔁 Lazy import the model logic here to prevent startup crashes
        from app.model_logic import run_full_pipeline

        # Run pipeline
        result_df = run_full_pipeline(file_url)

        # Return top 10 preview
        preview = result_df.head(10).to_dict(orient="records")
        logging.info("✅ Prediction successful")

        return JSONResponse(content=preview)

    except Exception as e:
        logging.error("🔥 Exception during prediction:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"},
        )

@app.get("/health")
def health_check():
    return {"status": "OK"}
