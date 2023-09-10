"""
The script runs BART model on zero shot classification task.
The model used could be found on: https://huggingface.co/facebook/bart-large-mnli

"""

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
print(classifier(sequence_to_classify, candidate_labels))

# output
#{'sequence': 'one day I will see the world', 
# 'labels': ['travel', 'dancing', 'cooking'], 
# 'scores': [0.9938650727272034, 0.003273803973570466, 0.0028610341250896454]}
