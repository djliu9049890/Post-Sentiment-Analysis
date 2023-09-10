"""
The script runs BART model on question answering task given a question and the answer in the text.
The model used could be found on: 
    https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForQuestionAnswering.forward.example

"""

from transformers import AutoTokenizer, BartForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-finetuned-squadv1")
model = BartForQuestionAnswering.from_pretrained("valhalla/bart-large-finetuned-squadv1")
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
#question, text = "What's the largest animal in the world?", "Elephant, ancient, friendly and largest mammal in the world."
inputs = tokenizer(question, text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)) # print " nice puppet" or "Elephant" for the 2nd question and text

print(answer_start_index) # tensor(14)
print(answer_end_index) # tensor(15)

# target is "nice puppet"

# compute loss
#target_start_index = torch.tensor([14]) # from above
#target_end_index = torch.tensor([15]) # from above
#outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
#loss = outputs.loss
#print(round(loss.item(), 2)) # 0.59