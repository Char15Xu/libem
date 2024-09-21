import libem
from benchmark.util import run_match


def run(train_set, test_set, args, model_choice="gpt-4o"):
    results, stats = {}, {}

    libem.calibrate({
        "libem.match.parameter.model": model_choice,
        "libem.match.parameter.confidence": True
    }, verbose=True)
    

    stats['match'], results['match'] = run_match(train_set, test_set, args)

    return stats, results