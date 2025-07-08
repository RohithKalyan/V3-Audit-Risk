import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.model_logic import run_full_pipeline  # ✅ Fixed: use correct function name

# ✅ Setup logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

class FileURLRequest(BaseModel):
    file_url: str

@app.post("/predict")
async def predict(request: FileURLRequest):
    try:
        file_url = request.file_url
        logging.debug(f"🔍 Received file URL: {file_url}")

        # ✅ Run your full model logic
        result_df = run_full_pipeline(file_url)

        # ✅ Return only top 10 rows for performance in web preview
        preview = result_df.head(10).to_dict(orient="records")

        logging.info("✅ /predict executed successfully")
        return JSONResponse(content=preview)

    except Exception as e:
        logging.error("🔥 Exception during /predict:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"},
        )

@app.get("/health")
def health_check():
    return {"status": "OK"}
