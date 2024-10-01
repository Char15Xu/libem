import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from libem.prepare.datasets import abt_buy
from libem.cascade.util import low_confidence_filter, profile
from libem.cascade.vectorize.function import run as vectorize
from libem.cascade.prematch.function import run as prematch
from libem.cascade.match.function import run as match


def online(args, dataset, prematch_model="gpt-4o-mini", match_model="gpt-4o", num_pairs=2, threshold=0.2):

    if dataset == abt_buy:
        train_set, test_set = vectorize(args, dataset, num_pairs)
    
        final_stats, final_result = {}, []
        prematch_stats, prematch_results = prematch(train_set, test_set, args, model_choice=prematch_model)
        
        unconfident_pairs, confident_pairs = low_confidence_filter(prematch_results, threshold)
        final_result += confident_pairs

        if unconfident_pairs:
            
            match_stats, match_results = match(train_set, unconfident_pairs, args, model_choice=match_model)
            final_result += match_results

        else:
            match_stats, match_results = {}, {}

    # Result when Compose is used
    overall_stat = profile(args, final_result, num_pairs)

    final_stats = {
        "stats": overall_stat, 
        "stats by stages": {
            f"Prematch {prematch_model}": prematch_stats,
            f"Match {match_model}": match_stats,
        },
        "results": final_result
        }
    
    # Result when only Prematch is used
    prematch_single_stats, prematch_single_results = prematch_stats, prematch_results
    
    prematch_single = {
        "stats": prematch_single_stats,
        "results": prematch_single_results
    }

    # Result when only Match is used
    match_single_state, match_single_result = match(train_set, test_set, args, model_choice=match_model)

    match_single = {
        "stats": match_single_state,
        "results": match_single_result
    }

    # Save the output
    output_path = os.path.join("examples", "cascade", "output")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Output of the Compose
    file_name = f"{timestamp}-abt-buy-compose.json"
    file_path = os.path.join(output_path, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(final_stats, json_file, indent=4)

    # Output of the Prematch
    file_name = f"{timestamp}-abt-buy-{prematch_model}.json"
    file_path = os.path.join(output_path, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(prematch_single, json_file, indent=4)

    # Output of the Match   
    file_name = f"{timestamp}-abt-buy-{match_model}.json"
    file_path = os.path.join(output_path, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(match_single, json_file, indent=4)

    # Plot the metrics comparison graph
    metrics_data = {'accuracy': {}, 'f1': {}, 'recall': {}, 'precision': {}}

    for model_name, data in zip([prematch_model, match_model, "Compose"], [prematch_single, match_single, final_stats]):
        metrics_data['accuracy'][model_name] = data['stats']['accuracy']
        metrics_data['f1'][model_name] = data['stats']['f1']
        metrics_data['recall'][model_name] = data['stats']['recall']
        metrics_data['precision'][model_name] = data['stats']['precision']

    approaches = list([prematch_model, match_model, "Compose"])
    metrics = ['precision', 'recall', 'f1', 'accuracy']

    metrics_values = [[metrics_data[metric][approach] for approach in approaches] for metric in metrics]

    bar_width = 0.2
    r = np.arange(len(metrics))
    offsets = [r + i * bar_width for i in range(len(approaches))]

    plt.figure(figsize=(10, 6))
    colors = ['dodgerblue', 'lightblue', 'skyblue']

    for i, approach in enumerate(approaches):
        plt.bar(offsets[i], [metrics_values[j][i] for j in range(len(metrics))], width=bar_width, color=colors[i], label=approach)

    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Comparison of Approaches across Metrics', fontsize=14)
    plt.xticks([r + bar_width for r in range(len(metrics))], metrics)
    plt.legend()
    plt.ylim(0, 100)

    plt.show()

    file_name = f"approaches-metrics-comparison-{timestamp}.png"
    file_path = os.path.join(output_path, file_name)
    plt.savefig(file_path)

    return final_stats, final_result


def offline(args):
    pass
