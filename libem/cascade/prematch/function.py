import libem
from libem.cascade.util import low_confidence_filter
from benchmark.util import run_block, run_match


def run(train_set, test_set, args, model_choice="gpt-4o-mini"):
    results, stats = {}, {}

    if model_choice == "block":
        test_set, stats['block'], results['block'] = run_block(test_set, args)

    else:
        libem.calibrate({
            "libem.match.parameter.model": model_choice,
            "libem.match.parameter.confidence": True
        }, verbose=True)
        
        stats['match'], results['match'] = run_match(train_set, test_set, args)

    return stats, results