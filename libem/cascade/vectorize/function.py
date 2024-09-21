import libem
import random


def run(dataset):
    random.seed(42)

    # construct kwargs dict
    kwargs = {
        'version': 1,
    }

    train_set = dataset.read_train(**kwargs)
    test_set = dataset.read_test(**kwargs)
    test_set = list(test_set)
    print(f"Number of test pairs: {len(test_set)}")
    test_set = test_set[:100]
    print(f"Number of test pairs: {len(test_set)}")
    random.shuffle(test_set)

    return train_set, test_set