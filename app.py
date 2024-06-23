from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import whisper
from transformers import pipeline
import os

app = FastAPI()

# Load models globally
model = whisper.load_model("large-v3")
summarizer = pipeline("summarization")


@app.get("/")
async def root():
    return {"message": "Welcome to the audio transcription and summarization API"}


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file locally
        audio_path = f"uploads/{file.filename}"
        with open(audio_path, "wb") as f:
            f.write(file.file.read())

        # Transcribe the audio
        result = model.transcribe(audio_path)

        # Extract the text and segments with timestamps
        transcription = result['text']
        segments = result['segments']

        return JSONResponse(content={"transcription": transcription, "segments": segments})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/")
async def summarize_text(text: str):
    try:
        summary = summarizer(text, max_length=150,
                             min_length=40, do_sample=False)
        return JSONResponse(content={"summary": summary[0]['summary_text']})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file locally
        audio_path = f"uploads/{file.filename}"
        with open(audio_path, "wb") as f:
            f.write(file.file.read())

        # Transcribe the audio
        result = model.transcribe(audio_path)

        # Extract the text and segments with timestamps
        transcription = result['text']
        segments = result['segments']

        # Summarize the transcribed text
        summary = summarizer(transcription, max_length=150,
                             min_length=40, do_sample=False)[0]['summary_text']

        # Save results to files
        transcription_path = f"results/{file.filename}_transcription.txt"
        summary_path = f"results/{file.filename}_summary.txt"

        os.makedirs(os.path.dirname(transcription_path), exist_ok=True)

        with open(transcription_path, "w") as f:
            f.write(transcription)

        with open(summary_path, "w") as f:
            f.write(summary)

        return JSONResponse(content={
            "transcription": transcription,
            "segments": segments,
            "summary": summary,
            "transcription_file": transcription_path,
            "summary_file": summary_path
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
