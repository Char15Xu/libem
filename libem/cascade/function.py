from libem.prepare.datasets import abt_buy
from libem.cascade.util import low_confidence_filter
from libem.cascade.vectorize.function import run as vectorize
from libem.cascade.prematch.function import run as prematch
from libem.cascade.match.function import run as match


def online(args, dataset):

    if dataset == abt_buy:

        train_set, test_set = vectorize(dataset)

        stats, results = prematch(train_set, test_set, args, model_choice="gpt-4o-mini")
        unconfident_set, confident_result = low_confidence_filter(results, threshold=0.1)

        if unconfident_set:
            print(f"Running Match")
            stats_match, results_match = match(train_set, unconfident_set, model_choice="gpt-4o")
            stats.append(stats_match)
            confident_result.append(results_match)

        print(f"Unconfident pairs: {unconfident_set}")
        print(f"Result: {results}")
        print(f"Stats: {stats}")

    return


def online(args):
    pass
