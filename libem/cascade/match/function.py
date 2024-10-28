import libem
import openai
import time
from libem.cascade.util import run as run_match


def run(train_set, test_set, args, model_choice="gpt-4o"):
    results, stats = {}, {}

    libem.calibrate({
        "libem.match.parameter.model": model_choice,
        "libem.match.parameter.confidence": True
    }, verbose=True)
    args.model = model_choice
    
    stats, results = run_match(train_set, test_set, args)

    print("args", args)
    print("run matching using model", args.model)
    
    return stats, results
    # max_retries = 3
    # retry_delay = 10
    # for attempt in range(max_retries):
    #     try:
    #         args.model = model_choice
    #         stats, results = run_match(train_set, test_set, args)
    #         return stats, results
    #     except openai.APIConnectionError as e:
    #         print(f"API connection error: {e}. Retrying in {retry_delay} seconds...")
    #         time.sleep(retry_delay)
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         break