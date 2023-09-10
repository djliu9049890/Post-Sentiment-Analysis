"""
The script is able to generate text given a starting phrase and text in English.

"""

from transformers import pipeline

generator = pipeline('text-generation', model = 'gpt2')

text = "Hello, I'm a language model"
print(generator(text, max_length = 30, num_return_sequences=3))


# print - produces different results at each run
#[{'generated_text': "Hello, I'm a language model!\n\nI want to get you interested in the world through my works. When I have time I will always"}, 
# {'generated_text': "Hello, I'm a language modeler.\n\nI hope that makes sense to you.\n\nThanks,\n\nSeth\n\nHi"}, 
# {'generated_text': "Hello, I'm a language modeler myself. But I don't want to say too much about it, but the only thing I've done is"}]
