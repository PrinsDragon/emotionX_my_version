import json

data_dir = "./EmotionPush/emotionpush_train.json"
data_file = open(data_dir, "r", encoding="utf-8")
data_json = json.load(data_file)

record = {}
emotion_name = [
        'neutral',
        'joy',
        'sadness',
        'anger',
        'disgust',
        'fear',
        'surprise',
        'non-neutral'
    ]

for dialog in data_json:
    for sentence in dialog:
        speaker = sentence["speaker"]
        emotion = sentence["emotion"]
        if speaker not in record:
            record[speaker] = {emotion_name[i]: 0 for i in range(8)}
        record[speaker][emotion] += 1

save_dir = "eval.txt"
save_file = open(save_dir, "w", encoding="utf-8")

for speaker in record:
    emotion = record[speaker]
    save_file.write("{};{};{};{};{}\n".format(speaker, emotion["neutral"], emotion["joy"], emotion["sadness"], emotion["anger"]))
