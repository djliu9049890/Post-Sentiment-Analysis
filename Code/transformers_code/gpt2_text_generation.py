"""
The script is able to generate text given a starting phrase and text in English.

"""

from transformers import AutoTokenizer, AutoModelForCausalLM

gpt2 = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(gpt2)
model = AutoModelForCausalLM.from_pretrained(gpt2)

#prompt = "Hugging Face Company is"
prompt = "Coding is"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=100)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# print - result is consistent from multiple runs
# ['Coding is a process of writing code.\n\nIt takes time to learn the language, 
# but once you do, your code becomes easier to reason about and more maintainable.\n\n
# The best way to learn is to use a tool like Codecademy, which has over 2,000 courses 
# for every skill you need.\n\n2. You need to be comfortable with JavaScript, HTML, CSS, 
# and other technologies that are part of the web development ecosystem.\n\nThis is the most important part of']