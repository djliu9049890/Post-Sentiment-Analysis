"""
The script runs BART model on sentiment classification task.
The model used could be found on: 
    https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForSequenceClassification.forward.example

"""

import torch
from transformers import AutoTokenizer, BartForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-sst2")
model = BartForSequenceClassification.from_pretrained("valhalla/bart-large-sst2")
#inputs = tokenizer("Hello, my dog is cute", return_tensors="pt") # one test input as string
inputs = tokenizer(["Hello, my dog is cute", "I don't like playing with it"], padding=True, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# support only one test input as a string
#print("Logits:", logits)
#predicted_class_id = logits.argmax().item()
#print(model.config.id2label[predicted_class_id]) # print "POSITIVE"

# support multiple test inputs in a list
predicted_class_ids = logits.argmax(dim=1).tolist()
for class_id in predicted_class_ids:
    print(model.config.id2label[class_id]) # print "POSITIVE" and "NEGATIVE"

# compute loss - one test input as a string only
# to train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
#num_labels = len(model.config.id2label)
#model = BartForSequenceClassification.from_pretrained("valhalla/bart-large-sst2", num_labels=num_labels)
#labels = torch.tensor([1])
#loss = model(**inputs, labels=labels).loss
#print(round(loss.item(), 2)) # print 0.0