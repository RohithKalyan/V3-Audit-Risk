import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.model_logic import run_model_logic  # model_logic.py should be inside the 'app' folder

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

        # Run prediction logic
        result_df = run_model_logic(file_url)

        # For now, limit to top 10 rows only
        preview = result_df.head(10).to_dict(orient="records")

        print("✅ FastAPI /predict endpoint hit")
        print("✅✅ FastAPI app is alive and has bound to port!")  # This is the Render port binding check

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
    print("✅ /health endpoint was hit — service is live.")
    return {"status": "OK"}
