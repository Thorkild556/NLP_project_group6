import os
# Suppress TensorFlow warnings/errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def evaluate(generations, grounds, device='cpu', batch_size=32):
    from bert_score import BERTScorer

    scorer = BERTScorer(model_type='bert-base-uncased', lang='en', device='cpu')
    P, R, F1 = scorer.score(generations, grounds)

    return P, R, F1

if __name__ == '__main__':
    generation = "i have bought an owl, but now it wants to fly away"
    ground = "an owl was bought, and it wants to fly to the grocery store"

    P, R, F1 = evaluate([generation], [ground], device='cpu')
    print(f"Precision: {P[0]:.4f}")
    print(f"Recall: {R[0]:.4f}")
    print(f"F1: {F1[0]:.4f}")