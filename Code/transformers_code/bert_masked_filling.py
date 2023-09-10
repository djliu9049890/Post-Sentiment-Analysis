"""
The script runs BERT model on masked filling task.
The model used could be found on: 
    https://huggingface.co/docs/transformers/v4.27.0/en/model_doc/bert#transformers.BertForMaskedLM.forward.example

"""

from transformers import AutoTokenizer, BertForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

print(tokenizer.decode(predicted_token_id)) # "paris"

labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

# mask labels of non-[MASK] tokens
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
outputs = model(**inputs, labels=labels)

print(round(outputs.loss.item(), 2)) # loss = 0.88