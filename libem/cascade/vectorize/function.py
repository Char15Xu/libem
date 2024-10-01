import libem
import random


def run(args, dataset, num_pairs=100):
    random.seed(42)

    # construct kwargs dict
    kwargs = {
        'version': 1,
    }

    args.num_pairs = num_pairs

    train_set = dataset.read_train(**kwargs)
    test_set = dataset.read_test(**kwargs)
    test_set = list(test_set)
    
    test_set = test_set[:num_pairs]
    print(f"Number of test pairs: {len(test_set)}")
    random.shuffle(test_set)

    return train_set, test_set