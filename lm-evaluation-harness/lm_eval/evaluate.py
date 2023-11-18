import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='chatgpt')
    parser.add_argument("--task", default='kobest_copa')
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--helm_evaluation", type=bool, default=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    ans=[]
    with open(f"{args.model}_{args.task}_{args.num_fewshot}-shot_results.jsonl", 'r') as rf:
        for line in rf:
            line = json.loads(line)
            if args.helm_evaluation:
                if line['answer'][0] in line['predict'][0]:
                    ans.append(1)
                else:
                    ans.append(0)
            else:
                if line['answer'] in line['predict']:
                    ans.append(1)
                else:
                    ans.append(0)

    print (sum(ans)/len(ans))