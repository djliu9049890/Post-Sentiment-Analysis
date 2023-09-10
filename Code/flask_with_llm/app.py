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
    text = "Hello, I'm a language model"
    response = gpt2_text_generation.gpt2_model(text)

    return response


if __name__ == "__main__":
  
    app.run(debug=True)