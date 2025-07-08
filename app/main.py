# === app/main.py ===
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
import os
from urllib.parse import urlparse, unquote
from app.model_logic import run_full_pipeline

app = FastAPI()

class FileInput(BaseModel):
    file_url: str

@app.post("/predict")
async def predict(input_data: FileInput):
    try:
        file_url = input_data.file_url

        # === STEP 1: Extract the filename from the URL ===
        parsed_url = urlparse(file_url)
        file_name = os.path.basename(parsed_url.path)
        file_name = unquote(file_name)
        file_path = os.path.join("test_files", file_name)

        # === STEP 2: Download the file from URL ===
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # === STEP 3: Run the full pipeline ===
        df_result = run_full_pipeline(file_path)
        return df_result.head(10).to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
