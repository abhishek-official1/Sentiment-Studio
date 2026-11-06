import json

nb = json.load(open('Explainable_Sentiment_Analysis_BERT_Research (1) (2).ipynb', 'r', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = c['source']
    first_line = src[0][:150].replace('\n', ' ') if src else '(empty)'
    print(f"Cell {i:3d} ({c['cell_type']:8s}): {first_line}")
