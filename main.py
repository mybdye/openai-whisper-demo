import whisper

model = whisper.load_model("tiny.en")
result = model.transcribe("audio2.mp3")
print(result["text"])