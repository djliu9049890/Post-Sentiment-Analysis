"""
The script runs BERT model on English sentence similarity task, where similarity sentences could be found.
The model used could be found on: https://huggingface.co/docs/hub/sentence-transformers#using-existing-models

"""

from sentence_transformers import SentenceTransformer, util

model_name = "sentence-transformers/all-MiniLM-L6-v2"

#sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer(model_name)
#embeddings = model.encode(sentences)
#print(embeddings)

query_embedding = model.encode('This is a happy person')
passage_embedding = model.encode(['This is a happy cat',
                                  'This is a very happy person',
                                  'Today is a sunny day'])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))

# output: Similarity: tensor([[0.6799, 0.9424, 0.2954]])