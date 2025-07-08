from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import traceback

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

class FileURLRequest(BaseModel):
    file_url: str

@app.post("/predict")
async def predict(request: FileURLRequest):
    try:
        from app.model_logic import run_model_logic  # Lazy import

        file_url = request.file_url
        logging.debug(f"🔍 Received file URL: {file_url}")
        result_df = run_model_logic(file_url)
        preview = result_df.head(10).to_dict(orient="records")
        print("✅ FastAPI /predict endpoint hit")
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
