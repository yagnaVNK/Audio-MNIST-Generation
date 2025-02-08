from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import librosa
import scipy
import time
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import uuid
from fastapi.staticfiles import StaticFiles
from src.Transformer import *
from src.TransformerMonai import *
from src.VQVAE_monai import *

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a directory for storing generated files
UPLOAD_DIR = "static/generated"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_device():
    """Get the appropriate device with error handling"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            return f'cuda:0'  # Always use the first available GPU
    return 'cpu'

def load_models():
    """Load models with proper device mapping"""
    try:
        VQVAE_PATH = 'saved_models/vqvae_monai.pth'
        MONAI_TRANSFORMER_MODEL_PATH = 'saved_models/MONAI_Cond2_Transformer_epochs_50.pt'
        TRANSFORMER_MODEL_PATH = 'saved_models/NanoGPT_Cond2_Transformer_epochs_50.pt'
        
        device = get_device()
        print(f"Using device: {device}")
        
        # Load models with explicit device mapping
        vqvae = torch.load(VQVAE_PATH, map_location=device)
        monai_model = torch.load(MONAI_TRANSFORMER_MODEL_PATH, map_location=device)
        gpt_model = torch.load(TRANSFORMER_MODEL_PATH, map_location=device)

        # Ensure models are on the correct device
        vqvae = vqvae.to(device)
        monai_model = monai_model.to(device)
        gpt_model = gpt_model.to(device)

        # Set to evaluation mode
        vqvae.eval()
        monai_model.eval()
        gpt_model.eval()

        return vqvae, monai_model, gpt_model, device
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

# Load models at startup
try:
    vqvae, monai_model, gpt_model, device = load_models()
except Exception as e:
    print(f"Failed to load models: {str(e)}")
    raise

class GenerationRequest(BaseModel):
    digit: int
    model: str

@app.post("/api/generate")
async def generate_audio(request: GenerationRequest):
    if not 0 <= request.digit <= 9:
        raise HTTPException(status_code=400, detail="Digit must be between 0 and 9")
    
    if request.model not in ["nanogpt", "monai"]:
        raise HTTPException(status_code=400, detail="Invalid model selection")

    try:
        # Time the indices generation
        start_time = time.time()
        BOS_TOKEN = 256
        context = torch.tensor([[BOS_TOKEN, 257 + request.digit]], dtype=torch.long, device=device)
        
        transformer_model = gpt_model if request.model == "nanogpt" else monai_model
        with torch.no_grad():  # Add no_grad for inference
            generated = transformer_model.generate(context, max_new_tokens=352)
            indices = generated[0, 2:]
            fake_indices = indices.view(1, 32, 11)
        indices_time = time.time() - start_time

        # Time the spectrogram generation
        start_time = time.time()
        with torch.no_grad():  # Add no_grad for inference
            fake_recon = vqvae.model.decode_samples(fake_indices)
            fake_recon_cpu = fake_recon[0].cpu().detach().numpy()
        
        tmp_tensor = fake_recon_cpu.reshape(2, fake_recon_cpu.shape[1]*fake_recon_cpu.shape[2])
        complex_data = tmp_tensor[0,:] + 1j * tmp_tensor[1,:]
        fake_recon_complex = complex_data.reshape(1, fake_recon_cpu.shape[1], fake_recon_cpu.shape[2])
        
        Img_fake = librosa.amplitude_to_db(np.abs(fake_recon_complex), ref=np.max)
        spec_time = time.time() - start_time

        # Time the audio generation
        start_time = time.time()
        _, audio = scipy.signal.istft(fake_recon_complex, 12000)
        audio_time = time.time() - start_time

        # Generate unique filename
        unique_id = str(uuid.uuid4())
        audio_filename = f"{unique_id}_audio.wav"
        spec_filename = f"{unique_id}_spec.png"

        # Save spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(Img_fake[0])
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Generated Spectrogram - Digit {request.digit}')
        plt.tight_layout()
        plt.savefig(os.path.join(UPLOAD_DIR, spec_filename))
        plt.close()

        # Save audio
        array = audio
        array = array / np.max(np.abs(array)) if np.max(np.abs(array)) > 0 else array
        array = (array * 32767).astype(np.int16)
        scipy.io.wavfile.write(os.path.join(UPLOAD_DIR, audio_filename), 12000, array.T)

        total_time = indices_time + spec_time + audio_time

        return JSONResponse({
            "times": {
                "indices": indices_time,
                "spectrogram": spec_time,
                "audio": audio_time,
                "total": total_time
            },
            "audioUrl": f"/static/generated/{audio_filename}",
            "spectrogramUrl": f"/static/generated/{spec_filename}"
        })

    except Exception as e:
        print(f"Error during generation: {str(e)}")  # Add logging
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8956)