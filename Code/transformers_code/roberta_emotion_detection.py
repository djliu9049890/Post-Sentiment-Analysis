"""
The script runs model on English sentiment classification task with three classes: positive, neutral and negative.
The model used could be found on: https://huggingface.co/SamLowe/roberta-base-go_emotions.
A simpler approach is also written below commented out.

"""

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig, pipeline
import numpy as np

MODEL = f"SamLowe/roberta-base-go_emotions"
emotion = pipeline(model=MODEL)

def top_five(text):
    emotion_labels = emotion(text, return_all_scores=True)
    emotion_labels = sorted(emotion_labels[0], key = lambda emotion: emotion['score'], reverse=True)[0:5]
    print(emotion_labels)
    for emotion in emotion_labels:
        print(emotion["label"], emotion["score"])
    return emotion_labels
#text = "Good night 😊"
#text = "Good night"
#text = "I love you!"