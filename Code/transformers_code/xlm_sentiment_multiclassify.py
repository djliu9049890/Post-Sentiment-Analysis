"""
The script runs model on English sentiment classification task with three classes: positive, neutral and negative.
The model used could be found on: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment.
A simpler approach is also written below commented out.

"""

import os.path
import shutil
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

"""
# Simpler approach
from transformers import pipeline
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
print(sentiment_task("T'estimo!"))
"""

MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

text = "Good night 😊"

encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

# Print labels and scores
ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = config.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")

# output
# 1) positive 0.7673
# 2) neutral 0.2015
# 3) negative 0.0313

if os.path.isdir("cardiffnlp"):
    # remove the cardiffnlp folder if exist
    shutil.rmtree("cardiffnlp")