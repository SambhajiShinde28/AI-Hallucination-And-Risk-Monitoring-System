from fastapi import FastAPI, File, UploadFile,requests
from Model_File import workflow
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(
    title="AI Hallucination And Risk Monitoring System",
    version="1.0.0",
    description="Hallucination detection and risk monitoring system for LLM outputs with real-time decision control."
)

class user_question_pydantic(BaseModel):
    ques:str
    file_path:str

@app.post("/ask/")
def user_question(req:user_question_pydantic):
    user_question=req.ques
    file_path=req.file_path

    output=workflow.invoke(
        {
            "PDFFile_path":file_path,
            "query":user_question
        }
    )

    return {
        "answer":output['llm_answer'].answer,
        "confidence":output['llm_answer'].confidence,
        "risk_level":output['llm_answer'].risk_level,
        "decision":output['llm_answer'].decision,
        "message":output['llm_answer'].warning_message
    }
    

@app.post("/file_upload/")
async def User_File_Uploaded(file: UploadFile = File(...)):
    folder_name = "User_Uploaded_Files"
    os.makedirs(folder_name, exist_ok=True)

    file_path = f"{folder_name}/{file.filename}"
    contents = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)

    return {"message": "File saved successfully.", "file_path": file_path}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
