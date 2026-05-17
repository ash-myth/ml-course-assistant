import os
import json
import whisper
model = whisper.load_model("medium", device="cuda")
os.makedirs("jsons", exist_ok=True)
audios = os.listdir("audios")
for audio in audios:
    if audio.endswith(".mp3"):
        parts = audio.split("_", 1)
        number = parts[0]
        title = parts[1][:-4] if len(parts) > 1 else audio[:-4]
        print(f"Processing: {number} - {title}")
        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        with open(f"jsons/{audio}.json", "w") as f:
            json.dump({"chunks": chunks, "text": result["text"]}, f)
        print(f"Done: {audio}")