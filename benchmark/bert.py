import os
import json
from datetime import datetime
from libem.prepare.datasets import abt_buy, amazon_google, beer, dblp_acm, dblp_scholar, fodors_zagats, itunes_amazon, walmart_amazon
from libem.match.interface.struct import parse_input

import torch
from transformers import RobertaTokenizer, RobertaModel

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt


_model, _tokenizer = None, None
_optimal_thresholds = None

def calculate_similarity(entity1, entity2):
    global _model, _tokenizer

    model_name = "roberta-base"
    device = "mps"

    if _model is None or _tokenizer is None:
        _tokenizer = RobertaTokenizer.from_pretrained(model_name)
        _model = RobertaModel.from_pretrained(model_name).to(device)
    else:
        pass

    model, tokenizer = _model, _tokenizer
    model.eval()

    # Tokenize and encode entities
    encoding1 = tokenizer.encode_plus(
        entity1,
        add_special_tokens=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    encoding2 = tokenizer.encode_plus(
        entity2,
        add_special_tokens=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        embedding1 = model(**encoding1).last_hidden_state.mean(dim=1)
        embedding2 = model(**encoding2).last_hidden_state.mean(dim=1)

    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

    return similarity


def get_similarity_scores(test_set):
    y_match = []
    similarities = []
    match_similarities = []
    no_match_similarities = []
    for i, data in enumerate(test_set):
        label = data["label"]
        left, right = data["left"], data["right"]
        entity1, entity2 = parse_input(left, right)
        y_match.append(label)
        similarity = calculate_similarity(entity1, entity2)
        similarities.append(similarity)
        if label:
            match_similarities.append(similarity)
        else:
            no_match_similarities.append(similarity)
        print(f"Processed No.{i + 1} \n Entity1: {entity1} \n Entity2: {entity2} \n Label: {label} \n Similarity: {similarities[-1]}")
    return y_match, similarities, match_similarities, no_match_similarities


def graph_pr(y_match, similarities, match_similarities, no_match_similarities, dataset):
    global _optimal_thresholds
    if _optimal_thresholds == None:
        _optimal_thresholds = {}
    optimal_thresholds = _optimal_thresholds

    # Plot Cosine Similarity Distributions
    plt.hist(match_similarities, bins=50, alpha=0.5, label='Matches')
    plt.hist(no_match_similarities, bins=50, alpha=0.5, label='Non-Matches')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Distribution of Cosine Similarities')
    output_dir = os.path.join('benchmark', '_output', 'results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'cosine_similarity_dist{dataset}_{timestamp}.png'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    # Save the figure
    plt.savefig(file_path)
    plt.close()

    # Plot Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_match, similarities)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    output_dir = os.path.join('benchmark', '_output', 'results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'precision_recall_curve_{dataset}_{timestamp}.png'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    # Save the figure
    plt.savefig(file_path)
    plt.close()

    # Find optimal F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall)

    optimal_idx = f1_scores.argmax()
    optimal_f1 = float(f1_scores[optimal_idx])
    optimal_threshold = float(thresholds[optimal_idx])

    opt_result = {"f1_scores": optimal_f1, "optimal_threshold": optimal_threshold}
    optimal_thresholds[dataset] = opt_result
    _optimal_thresholds = optimal_thresholds


def bert_analyze(dataset):
    test_set = dataset.read_test()
    dataset_name = dataset.__name__.split('.')[-1]

    y_match, similarities, match_similarities, no_match_similarities = get_similarity_scores(test_set)
    graph_pr(y_match, similarities, match_similarities, no_match_similarities, dataset_name)


def reset():
    global _model, _tokenizer
    _model, _tokenizer = None, None


if __name__ == '__main__':
    datasets = [abt_buy, amazon_google, beer, dblp_acm, dblp_scholar, 
                fodors_zagats, itunes_amazon, walmart_amazon]

    for dataset in datasets:
        bert_analyze(dataset)
        reset()

    _optimal_thresholds_json = json.dumps(_optimal_thresholds, indent=4)
    print(_optimal_thresholds_json)