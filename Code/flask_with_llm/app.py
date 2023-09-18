"""
conda activate flask or an env that have both flask AND transformers (HuggingFace) libraries installed.
In the root directory, run `python app.py` and navigate to http://127.0.0.1:5000 to see output of the GPT-2 model.

"""

from flask import Flask

from hello_world import hello
from hello_world import gpt2_text_generation

app = Flask("abc")

@app.route('/')
def index():

    # response = hello.hello()
    #topic = "However, all states have their own laws, enforcement mechanisms, and prescribed punishments for breaking the laws; so, it seems people are generally in favor of some surveillance. What are your thoughts? Concerning behavior engineering (e.g., social media sites), what are your thoughts?"
    #text = "I think for people, there should be some sort of surveillance for public safety. Of course, there should be some sort of balance. I think strong surveillance to the extent that everyone in the society feels that it's excessive is wrong. However, even in a society where some surveillance exists there are crimes. I think using artificial intelligence maybe for the purpose of tracking potential felonies or terrorists should be allowed for public safety. However, AI usage in commercial benefits especially its usage in internet advertisement, I think it should be more debated, and I personally am negative about it. In conclusion, I agree with AI surveillance systems for the betterment of the public, but if it is for the profit of individual or private companies there should be some limit in terms of whether individual or private companies can track and surveil other people's activity on internet or just in general."
    topic = "hi"
    text = "i love eating french fries"
    response = gpt2_text_generation.gpt2_model(topic, text)

    return response


if __name__ == "__main__":
  
    app.run(debug=True)