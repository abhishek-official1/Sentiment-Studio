import json

nb = json.load(open('Explainable_Sentiment_Analysis_BERT_Research (1) (2).ipynb', 'r', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        full_src = ''.join(c['source'])
        # Find output file references
        for line in full_src.split('\n'):
            if any(kw in line for kw in ['write_text', 'to_csv', 'savefig', 'save_pretrained', 'torch.save', 'json.dump', 'to_json', 'OUT', 'FIG']):
                print(f"Cell {i:3d}: {line.strip()[:150]}")
