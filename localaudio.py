import edge_tts
import asyncio
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    voice: str
    rate: int = 0   # Speech rate adjustment in percentage (default: 0)
    pitch: int = 0  # Pitch adjustment in Hz (default: 0)

async def text_to_speech(text: str, voice: str, rate: int = 0, pitch: int = 0) -> str:
    """
    Converts the provided text to speech and saves it as an MP3 file.

    Parameters:
        text (str): The text to convert.
        voice (str): The voice identifier. If provided in a formatted string 
                     (e.g., "Jenny - en-US (Female)"), the function extracts the short name.
        rate (int): Rate adjustment in percentage.
        pitch (int): Pitch adjustment in Hz.

    Returns:
        str: The path to the saved MP3 file.
    """
    if not text.strip():
        raise ValueError("Please enter text to convert.")
    if not voice:
        raise ValueError("Please select a voice.")

    # Extract the short name if the voice is provided in a formatted string.
    voice_short_name = voice.split(" - ")[0]
    rate_str = f"{rate:+d}%"
    pitch_str = f"{pitch:+d}Hz"

    # Initialize the Edge TTS communicator.
    communicate = edge_tts.Communicate(text, voice_short_name, rate=rate_str, pitch=pitch_str)

    # Create a temporary file to store the audio.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
        # Await the saving of the file; this ensures the file is complete.
        await communicate.save(tmp_path)
    
    return tmp_path

@app.post("/synthesize")
async def synthesize_tts(tts_request: TTSRequest):
    """
    Endpoint to convert text to speech.

    Request Body (JSON):
    {
        "text": "Your text here",
        "voice": "en-US-JennyNeural",  // or a formatted string like "Jenny - en-US (Female)"
        "rate": 0,
        "pitch": 0
    }

    Returns:
        An MP3 file as the response.
    """
    try:
        audio_path = await text_to_speech(
            tts_request.text, 
            tts_request.voice, 
            tts_request.rate, 
            tts_request.pitch
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Create a generator to stream the file content.
    def iterfile():
        with open(audio_path, mode="rb") as file_like:
            yield from file_like

    # Return the file as a streaming response with a content-disposition header.
    return StreamingResponse(
        iterfile(), 
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=output.mp3"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
