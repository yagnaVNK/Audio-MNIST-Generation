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
import uuid
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/generated"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_device():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            return f'cuda:0'
    return 'cpu'

def load_models():
    try:
        VQVAE_PATH = 'saved_models/vqvae_monai.pth'
        MONAI_COND_PATH = 'saved_models/MONAI_Cond2_Transformer_epochs_50.pt'
        NANOGPT_COND_PATH = 'saved_models/NanoGPT_Cond2_Transformer_epochs_50.pt'
        MONAI_UNCOND_PATH = 'saved_models/MONAI_Transformer_epochs_50.pt'
        NANOGPT_UNCOND_PATH = 'saved_models/NanoGPT_Transformer_epochs_50.pt'
        
        device = get_device()
        print(f"Using device: {device}")
        
        vqvae = torch.load(VQVAE_PATH, map_location=device)
        monai_cond = torch.load(MONAI_COND_PATH, map_location=device)
        gpt_cond = torch.load(NANOGPT_COND_PATH, map_location=device)
        monai_uncond = torch.load(MONAI_UNCOND_PATH, map_location=device)
        gpt_uncond = torch.load(NANOGPT_UNCOND_PATH, map_location=device)

        models = {
            'conditional': {
                'monai': monai_cond.to(device),
                'nanogpt': gpt_cond.to(device)
            },
            'unconditional': {
                'monai': monai_uncond.to(device),
                'nanogpt': gpt_uncond.to(device)
            }
        }

        vqvae = vqvae.to(device)
        
        # Set all models to eval mode
        vqvae.eval()
        for model_type in models.values():
            for model in model_type.values():
                model.eval()

        return vqvae, models, device
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

try:
    vqvae, models, device = load_models()
except Exception as e:
    print(f"Failed to load models: {str(e)}")
    raise

class GenerationRequest(BaseModel):
    digit: int | None = None
    model: str
    model_type: str

@app.post("/api/generate")
async def generate_audio(request: GenerationRequest):
    if request.model_type not in ["conditional", "unconditional"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    if request.model not in ["nanogpt", "monai"]:
        raise HTTPException(status_code=400, detail="Invalid model selection")
    
    if request.model_type == "conditional":
        if request.digit is None or not 0 <= request.digit <= 9:
            raise HTTPException(status_code=400, detail="Digit must be between 0 and 9 for conditional models")

    try:
        start_time = time.time()
        BOS_TOKEN = 256
        
        if request.model_type == "conditional":
            context = torch.tensor([[BOS_TOKEN, 257 + request.digit]], dtype=torch.long, device=device)
        else:
            context = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)
        
        transformer_model = models[request.model_type][request.model]
        with torch.no_grad():
            generated = transformer_model.generate(context, max_new_tokens=352)
            indices = generated[0, 2:] if request.model_type == "conditional" else generated[0, 1:]
            fake_indices = indices.view(1, 32, 11)
        indices_time = time.time() - start_time

        # Generate spectrogram
        start_time = time.time()
        with torch.no_grad():
            fake_recon = vqvae.model.decode_samples(fake_indices)
            fake_recon_cpu = fake_recon[0].cpu().detach().numpy()
        
        tmp_tensor = fake_recon_cpu.reshape(2, fake_recon_cpu.shape[1]*fake_recon_cpu.shape[2])
        complex_data = tmp_tensor[0,:] + 1j * tmp_tensor[1,:]
        fake_recon_complex = complex_data.reshape(1, fake_recon_cpu.shape[1], fake_recon_cpu.shape[2])
        
        Img_fake = librosa.amplitude_to_db(np.abs(fake_recon_complex), ref=np.max)
        spec_time = time.time() - start_time

        # Generate audio
        start_time = time.time()
        _, audio = scipy.signal.istft(fake_recon_complex, 12000)
        audio_time = time.time() - start_time

        unique_id = str(uuid.uuid4())
        #audio_filename = f"{unique_id}_audio.wav"
        #spec_filename = f"{unique_id}_spec.png"
        audio_filename = f"sample_audio.wav"
        spec_filename = f"sample_spec.png"

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(Img_fake[0])
        plt.colorbar(format='%+2.0f dB')
        title = f'Generated Spectrogram - {"Digit " + str(request.digit) if request.model_type == "conditional" else "Unconditional"}'
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(UPLOAD_DIR, spec_filename))
        plt.close()

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
        print(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8956)