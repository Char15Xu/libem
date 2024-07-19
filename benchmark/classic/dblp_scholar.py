import json
import random

import libem
from libem.core.struct import Prompt
from libem.prepare.datasets import dblp_scholar

from benchmark import util
from benchmark.classic import block_similarities


def run(args):
    '''
    kwargs:
        version (int): the version of the dataset to use.
        keep_null (bool): if False, replace null values with empty str, else keep as 'None'.
        fields (list[str]): fields (and their order) to include in the output, 
                            empty to include all fields. Do not include _left/_right.
        domain_prompt (bool): if True, modifies the prompt to be domain-specific.
    '''
    random.seed(args.seed)

    # construct kwargs dict
    kwargs = {
        'schema': args.schema,
        'version': 1
    }
    if args.schema:
        kwargs['keep_null'] = True
        kwargs['fields'] = ["title", "authors", "venue", "year"]
    else:
        kwargs['keep_null'] = False
        kwargs['fields'] = ["authors", "title", "venue", "year"]

    kwargs.update(args.kwargs or {})

    # get dataset with kwargs
    if args.train:
        dataset = list(dblp_scholar.read_train(**kwargs))
    else:
        dataset = list(dblp_scholar.read_test(**kwargs))
    if args.shuffle:
        random.shuffle(dataset)

    # set domain prompt
    if 'domain_prompt' in kwargs and kwargs['domain_prompt'] is True:
        libem.calibrate({
            "libem.match.prompt.query": "Do the two publications refer to the same real-world publication? "
                                        "Answer with 'Yes' if they do and 'No' if they do not.\n"
                                        "Publication 1: '{left}'\nPublication 2: '{right}'",
            "libem.match.prompt.rules": Prompt.Rules(),
            "libem.match.prompt.experiences": Prompt.Experiences(),
            "libem.match.prompt.output": ""
        })

    if args.block:
        libem.calibrate({
            "libem.block.parameter.similarity":
                args.similarity
                if 0 <= args.similarity <= 100
                else block_similarities['dblp-scholar']
        })

        left = set(json.dumps(d['left']) for d in dataset)
        right = set(json.dumps(d['right']) for d in dataset)
        dataset = {
            'left': [json.loads(i) for i in left],
            'right': [json.loads(i) for i in right],
            'true': [{'left': d['left'], 'right': d['right']}
                     for d in dataset if d['label'] == 1]
        }

    return util.benchmark(dataset, args)
