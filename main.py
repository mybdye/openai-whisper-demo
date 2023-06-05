import whisperx, os

YOUR_HF_TOKEN = os.environ['YOUR_HF_TOKEN']
device = "cuda" 

audio_file = "amzn-captcha-modal.mp3"
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device, num_speakers=2)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio_file)
# diarize_model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs
