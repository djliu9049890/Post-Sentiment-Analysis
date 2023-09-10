"""
The script runs BART model on text summarization task.
The model used could be found on: 
    https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration.forward.example

"""

from transformers import AutoTokenizer, BartForConditionalGeneration

# read two full paragraphs and test BART summarization method
file = open("text/bert.txt", "r")
bert = file.read()
file.close()

file = open("text/wildfire.txt", "r")
wildfire = file.read()
file.close()

# summarize the two paragraphs
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
inputs = tokenizer([bert, wildfire], padding=True, truncation=True, max_length=512, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=100)

print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]) # output summary on bert
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[1]) # output summary on wildfire