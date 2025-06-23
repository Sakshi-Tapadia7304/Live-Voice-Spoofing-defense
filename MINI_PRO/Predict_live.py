import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from models.model import CNNNet
import librosa
import torch.nn.functional as F

DURATION = 5
SAMPLE_RATE = 16000
MAX_LEN = 256
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
THRESHOLD = 0.01

print("🎙 Listening...")
recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

volume = np.linalg.norm(recording)
print(f"🎚 Volume: {volume:.4f}")

if volume < THRESHOLD:
    print("😶 Too quiet. Exiting.")
    exit()

write("input.wav", SAMPLE_RATE, recording)

try:
    y, sr = librosa.load("input.wav", sr=SAMPLE_RATE)
except Exception as e:
    print(f"⚠ Failed to load audio: {e}")
    exit()

y = np.clip(y, -1.0, 1.0)

mel = librosa.feature.melspectrogram(
    y=y, sr=sr,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS
)

log_mel = librosa.power_to_db(mel, ref=np.max)
mel_tensor = torch.tensor(log_mel).unsqueeze(0).float()

if mel_tensor.shape[-1] > MAX_LEN:
    mel_tensor = mel_tensor[:, :, :MAX_LEN]
else:
    mel_tensor = F.pad(mel_tensor, (0, MAX_LEN - mel_tensor.shape[-1]))

mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-6)
input_tensor = mel_tensor.unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNNet().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    predicted = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted].item()
label_map = {0: "❌ Spoofed Voice", 1: "✅ Real Voice"}
print(f"\n🧠 Prediction: {label_map[predicted]} (Confidence: {confidence:.2f})")