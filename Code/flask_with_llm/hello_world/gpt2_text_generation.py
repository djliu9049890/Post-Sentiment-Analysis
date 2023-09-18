"""
The script is able to generate text given a starting phrase and text in English.

"""

from transformers import pipeline

def gpt2_model(topic, text):

    generator = pipeline("text-generation", model = "gpt2")

    output = generator("Can you provide a prompt that counters the user's argument and challenges" +  
                       " the user to think about a different perspective to: " + topic + 
                       " This is the user's argument: " + text
                       , max_length = 100, num_return_sequences=3)

    print(output)

    return output


# print - produces different results at each run
#[{'generated_text': "Hello, I'm a language model!\n\nI want to get you interested in the world through my works. When I have time I will always"}, 
# {'generated_text': "Hello, I'm a language modeler.\n\nI hope that makes sense to you.\n\nThanks,\n\nSeth\n\nHi"}, 
# {'generated_text': "Hello, I'm a language modeler myself. But I don't want to say too much about it, but the only thing I've done is"}]
