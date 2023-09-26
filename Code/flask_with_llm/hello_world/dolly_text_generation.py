import torch
from transformers import pipeline

def generate(topic, comment):
    #"t5-small"
    #"EleutherAI/gpt-neo-125M"
    #"Arjun-G-Ravi/chat-GPT2"
    #"Qiliang/bart-large-cnn-samsum-ChatGPT_v3"
    generator = pipeline("text-generation", model="chkla/roberta-argument")
    text = f"Read the question and give an honest answer. Your answers should not include any unethical, racist, sexist, dangerous, or illegal content. If the question is wrong, or does not make sense, accept it instead of giving the wrong answer.\nQuestion: Can you argue against this comment: '{comment}'?\nAnswer: "
    output = generator(text, max_length = 200, num_return_sequences=1)
    return output

# def generate(topic, comment):
#     generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
#     text = "Can you provide a prompt that counters the user's argument and challenges"+ " the user to think about a different perspective for: " + topic + " This is the user's argument: " + comment
#     res = generate_text(text)
#     return res[0]["generated_text"]

