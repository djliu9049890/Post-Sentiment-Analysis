"""
The script runs BART model on sentiment classification task with multiple labels for a given input.
The model used could be found on: 
    https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForSequenceClassification.forward.example-2

"""

import torch
from transformers import AutoTokenizer, BartForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-sst2")
model = BartForSequenceClassification.from_pretrained("valhalla/bart-large-sst2", problem_type="multi_label_classification")
#inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
inputs = tokenizer("I don't like playing with it", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

# to train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = BartForSequenceClassification.from_pretrained(
    "valhalla/bart-large-sst2", num_labels=num_labels, problem_type="multi_label_classification"
)

labels = torch.sum(
    torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
).to(torch.float)

print("Labels:", labels)

loss = model(**inputs, labels=labels).loss
print("Loss:", loss)

predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id]) # print "NEGATIVE"