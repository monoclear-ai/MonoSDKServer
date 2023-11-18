import sys
sys.path.append('..')

import json
from lm_eval.models.chatgpt import ChatGPT
from lm_eval.models.gpt3 import GPT3LM
from lm_eval import tasks
import lm_eval.tasks
import argparse
import random
import collections
import itertools
import fnmatch

davinci_models = ['davinci', 'text-davinci-002', 'text-davinci-003']
# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='chatgpt')
    parser.add_argument("--task", default='kobest_boolq')
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=9999999999)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    if args.model == 'chatgpt':
        model = ChatGPT()
    elif 'davinci' in args.model:
        if args.model in davinci_models:
            model = GPT3LM(engine=args.model, load_tokenizer=False)
        else:
            raise NotImplementedError("The model is not implemented")
    else:
        raise NotImplementedError("The model is not implemented")
    
    task_names = pattern_match(args.task.split(","), tasks.ALL_TASKS)
    task_dict = lm_eval.tasks.get_task_dict(task_names)

    task_dict_items = [
                        (name, task)
                        for name, task in task_dict.items()
                        if (task.has_validation_docs() or task.has_test_docs())
                        ]
    docs = collections.defaultdict(list)
    versions = collections.defaultdict(dict)
    requests = collections.defaultdict(list)
    labels = collections.defaultdict(list)


    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(args.seed)
        rnd.shuffle(task_docs)

        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, None)):
            ctx = task.fewshot_context(
                doc=doc, num_fewshot=args.num_fewshot, rnd=rnd,)
            req = task.construct_ko_text(doc, ctx)
            requests[task_name].append(req)
            labels[task_name].append(task.doc_to_target(doc).strip())
            docs[task_name].append(doc)
            
    nums=0
    for task_name, task in task_dict_items:
        with open(f"{args.model}_{task_name}_{args.num_fewshot}-shot_results.jsonl", 'wt') as wf:
            for req, label, doc in zip(requests[task_name], labels[task_name], docs[task_name]):
                if nums > args.limit:
                    break
                if args.model == 'chatgpt':
                    output = model.get_result(req)
                    ans_dict = {}
                    ans_dict['text'] = req
                    ans_dict['answer'] = label
                    if 'choices' in doc.keys():
                        ans_dict['choices'] = doc['choices']
                    else:
                        ans_dict['choices'] = None

                    try:
                        ans_dict['predict'] = output['choices'][0]['message']['content'].strip()
                    except:
                        ans_dict['predict'] = None

                wf.write(json.dumps(ans_dict, ensure_ascii=False))
                wf.write('\n')
                nums += 1
