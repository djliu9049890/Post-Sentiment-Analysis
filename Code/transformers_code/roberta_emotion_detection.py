"""
The script runs model on English sentiment classification task with three classes: positive, neutral and negative.
The model used could be found on: https://huggingface.co/SamLowe/roberta-base-go_emotions.
A simpler approach is also written below commented out.

"""

import os.path
import shutil
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig, pipeline
import numpy as np
#from scipy.special import softmax

"""
# Simpler approach
from transformers import pipeline
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
print(sentiment_task("T'estimo!"))
"""

MODEL = f"SamLowe/roberta-base-go_emotions"
emotion = pipeline(model=MODEL)

#text = "Good night 😊"
text = "I love you!"

emotion_labels = emotion(text, return_all_scores=True)
emotion_labels = sorted(emotion_labels[0], key = lambda emotion: emotion['score'], reverse=True)[0:5]
print(emotion_labels)
for emotion in emotion_labels:
    print(emotion["label"], emotion["score"])

if os.path.isdir("SamLowe"):
    # remove the SamLowe folder if exist
    shutil.rmtree("SamLowe")