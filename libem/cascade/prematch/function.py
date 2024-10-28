import libem
from libem.cascade.util import run as run_prematch
from libem.optimize import profile
from benchmark.util import run_block

def run(train_set, test_set, args, model_choice="gpt-4o-mini"):
    results, stats = {}, {}

    if model_choice == "block":
        test_set, stats['block'], results['block'] = run_block(test_set, args)

    else:
        libem.calibrate({
            "libem.match.parameter.model": model_choice,
            "libem.match.parameter.confidence": True
        }, verbose=True)
        args.model = model_choice
        print("args.model", args.model)
        stats, results = run_prematch(train_set, test_set, args)
    
    return stats, results