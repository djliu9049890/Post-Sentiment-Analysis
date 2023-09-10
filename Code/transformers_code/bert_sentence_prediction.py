"""
The script runs BERT model on English next sentence prediction to determine if the next sentence is likely random 
from the previous sentence.
The model used could be found on: https://huggingface.co/docs/transformers/v4.27.0/en/model_doc/bert#transformers.BertForNextSentencePrediction.forward.example

"""

from transformers import AutoTokenizer, BertForNextSentencePrediction
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."

encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

outputs = model(**encoding, labels=torch.LongTensor([1]))
logits = outputs.logits

print(logits[0, 0]) # -3.0729
print(logits[0, 1]) # 5.9056
assert logits[0, 0] < logits[0, 1]  # next sentence was random