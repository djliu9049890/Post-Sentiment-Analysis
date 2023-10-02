"""
conda activate flask or an env that have both flask AND transformers (HuggingFace) libraries installed.
In the root directory, run `python app.py` and navigate to http://127.0.0.1:5000 to see output of the GPT-2 model.

"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from hello_world import hello
from hello_world import gpt2_text_generation
from hello_world import emotion_detection
import os, logging

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.DEBUG)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/counter', methods=['POST'])
def counter():

    # response = hello.hello()
    #topic = "However, all states have their own laws, enforcement mechanisms, and prescribed punishments for breaking the laws; so, it seems people are generally in favor of some surveillance. What are your thoughts? Concerning behavior engineering (e.g., social media sites), what are your thoughts?"
    #text = "I think for people, there should be some sort of surveillance for public safety. Of course, there should be some sort of balance. I think strong surveillance to the extent that everyone in the society feels that it's excessive is wrong. However, even in a society where some surveillance exists there are crimes. I think using artificial intelligence maybe for the purpose of tracking potential felonies or terrorists should be allowed for public safety. However, AI usage in commercial benefits especially its usage in internet advertisement, I think it should be more debated, and I personally am negative about it. In conclusion, I agree with AI surveillance systems for the betterment of the public, but if it is for the profit of individual or private companies there should be some limit in terms of whether individual or private companies can track and surveil other people's activity on internet or just in general."
    topic = "hi"
    text = "i love eating french fries"
    response = gpt2_text_generation.gpt2_model(topic, text)
    return response

@app.route('/emotion', methods=['GET', 'POST'])
def emotion():
    # if request.method == 'GET':
        # app.logger.debug("hehllo")
        # print("hello")
        # print(os.getcwd())
        # return render_template('index.html')
    if request.method == 'POST':
        # print("hello")
        comment = request.json['comment']
        response = emotion_detection.top_five(comment)
        new_response = []
        for item in response:
            new_item = {
                "label": item["label"],
                "score": item["score"]
            }
            new_response.append(new_item)
        return new_response


if __name__ == "__main__":
    app.run(port=5001, debug=True)