# === FINAL main.py with all fixes applied ===
import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import pandas as pd

# Setup logging
import os
os.environ["NUMBA_LOG_LEVEL"] = "WARNING"
import logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Enable CORS for Swagger UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileURLRequest(BaseModel):
    file_url: str

@app.post("/predict")
async def predict(request: FileURLRequest):
    logging.debug("Received /predict request")
    try:
        file_url = request.file_url
        logging.debug(f"🔍 Received file URL: {file_url}")

        # === Download the file to a local path ===
        os.makedirs("test_files", exist_ok=True)
        local_path = "test_files/temp_input.csv"

        response = requests.get(file_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch file. Status code: {response.status_code}. URL: {file_url}")

        with open(local_path, "wb") as f:
            f.write(response.content)

        # === Lazy import of model logic ===
        from app.model_logic import run_full_pipeline

        logging.debug("Calling run_full_pipeline")
        # === Run Pipeline ===
        result_df = run_full_pipeline(local_path)
        logging.debug("run_full_pipeline returned")

        # === Fix datetime serialization issue ===
        for col in result_df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
            result_df[col] = result_df[col].astype(str)

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
