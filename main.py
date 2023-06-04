import whisper

model = whisper.load_model("tiny.en")
result = model.transcribe("amzn-captcha-modal.mp3")
print(result["text"])
