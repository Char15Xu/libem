import time
from libem.cascade.util import low_confidence_filter
from libem.cascade.vectorize.function import run as vectorize
from libem.cascade.prematch.function import run as prematch
from libem.cascade.match.function import run as match


def online(args, dataset, prematch_model="gpt-4o-mini", match_model="gpt-4o", num_pairs=2, threshold=0.2):
    args.sync = True

    if dataset:
        train_set, test_set = vectorize(args, dataset, num_pairs)
    
        cascade_stats, cascade_result = {}, []
        start_time = time.time()
        prematch_stats, prematch_results = prematch(train_set, test_set, args, model_choice=prematch_model)
        
        unconfident_pairs, confident_pairs = low_confidence_filter(prematch_results, threshold)
        cascade_result += confident_pairs

        if unconfident_pairs:
            match_stats, match_results = match(train_set, unconfident_pairs, args, model_choice=match_model)
            end_time = time.time()
            cascade_result += match_results

        else:
            match_stats, match_results = {}, {}
            end_time = time.time()
    

    results_data = {
            "args": args,
            "train_set": train_set,
            "test_set": test_set,
            "prematch_model": prematch_model,
            "match_model": match_model,
            "num_pairs": num_pairs,
            "threshold": threshold,
            "prematch_stats": prematch_stats,
            "prematch_results": prematch_results,
            "match_stats": match_stats,
            "match_results": match_results,
            "cascade_stats": cascade_stats,
            "cascade_result": cascade_result,
            "start_time": start_time,
            "end_time": end_time
        }
        
    return results_data


def offline(args):
    pass
