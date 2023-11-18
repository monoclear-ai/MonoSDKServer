import itertools
import random
import uuid
from datetime import datetime
from typing import Dict

from fastapi import BackgroundTasks, FastAPI
import collections
import lm_eval.tasks

from utils.connect.db_samples import db_samples
from utils.connect.models import Ranking
from utils.message import ClientMessage, Identifier, ClientAction, Task, ServerMessage, ServerAction, TaskPayload
from utils.serverconfig import RunConfig
from utils.connect.db_evals import db_evals
from utils.connect.db_ranking import db_ranking

from huggingface_hub.hf_api import HfFolder

# Token to test korean.
HfFolder.save_token('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')

app = FastAPI()

db_evals.init()
db_ranking.init()
db_samples.init()


class PerIdObject:
    def __init__(self, id: Identifier):
        self.id = id
        self.msg_queue = collections.deque()

        self.versions = collections.defaultdict(dict)

        self.requests = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)

        self.docs = collections.defaultdict(list)

        self.matches = collections.defaultdict(int)
        self.evals = collections.defaultdict(int)

        self.samples = collections.defaultdict(list)


TASK_ID = {
    Task.HAERAE.value: "haerae"
}

TASK_LISTS = {
    # Task.KOBEST.value: ["kobest_boolq", "kobest_copa", "kobest_hellaswag", "kobest_sentineg", "kobest_wic"],
    Task.HAERAE.value: ["haerae_hi", "haerae_kgk", "haerae_lw", "haerae_rc", "haerae_rw", "haerae_sn"],
    # Task.KLUE.value: ["klue_nli", "klue_sts", "klue_ynat"]
}

# TODO : Queue TTL (Time To Live)
objects = {}


def run_eval_loop(id: Identifier, task: Task, runConfig: RunConfig = RunConfig(num_fewshot=3)):
    # TODO : create server queue

    print(f"{task} started, ID:{id}, RunConfig: {runConfig}")
    # New ID
    obj = PerIdObject(id=id)
    objects[id] = obj

    task_list = TASK_LISTS[task]
    task_dict = lm_eval.tasks.get_task_dict(task_list)

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    obj.msg_queue.append(ServerMessage(action=ServerAction.START, id=id))

    # get lists of each type of request
    for task_name, task in task_dict_items:
        obj.versions[task_name] = task.VERSION
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)
        print(f"Task: {task_name}; number of docs: {len(task_docs)}")

        # Unpack config
        num_fewshot = runConfig.num_fewshot
        limit = runConfig.limit

        if limit is not None:
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
            ctx = task.fewshot_context(doc=doc, num_fewshot=num_fewshot, rnd=rnd, )
            req = task.construct_ko_text(doc, ctx)
            obj.requests[task_name].append(req)
            obj.labels[task_name].append(task.doc_to_target(doc).strip())
            obj.docs[task_name].append(doc)

            # Add to queue
            obj.msg_queue.append(ServerMessage(action=ServerAction.INPUT, id=id,
                                               data={"idx": doc_id,
                                                     "task_id": task_name,
                                                     "prompt": req}))
    # TODO : Purge queue on disconnect
    # TODO : Last message handling properly (at ACK time)
    obj.msg_queue.append(ServerMessage(action=ServerAction.END, id=id))


def _parse_upload(result: Dict) -> Dict:
    res = {}
    for task, output in result.items():
        res[task] = output["accuracy"]
    return res


def _upload(res, samples, eval_key, model_tag, task):
    payload = _parse_upload(res)
    print(payload)

    submit_at = datetime.now().isoformat()

    eval = db_evals.get(eval_key=eval_key)

    update_eval = {
        'model_tags': eval['model_tags'],
        'perf_score': eval['perf_score'],
        # 'perf_mistakes': eval['perf_mistakes'],
        'submit_dates': eval['submit_dates'],
        'sample_links': eval['sample_links'] if 'sample_links' in eval else []
    }
    base_tag = model_tag
    cnt = 0
    while model_tag in update_eval['model_tags']:
        print(f"Model tag '{model_tag}' is already uploaded.")
        model_tag = f"{base_tag}-{cnt}"
        cnt += 1
    p_idx = 1
    update_eval['model_tags'].append(model_tag)
    update_eval['submit_dates'].append(submit_at)
    sample_key = f"{eval_key}__{model_tag}__{p_idx}"
    # FIXME : Proper sample handling
    db_samples.create(sample_key, {
        'sample_key': sample_key,
        'sample_type': "wrong_only",
        'samples': samples
    })
    update_eval['sample_links'].append(f"/backend/eval/{eval_key}/samples/{model_tag}/1")

    if model_tag not in update_eval['perf_score']:
        update_eval['perf_score'][model_tag] = {}
    update_eval['perf_score'][model_tag][task] = payload
    print(update_eval)

    eval_res = db_evals.update(eval_key, update_eval)
    print(f"Eval result : {eval_res}")

    # Only add to private ranking on the same condition as email - when a set of tasks are completed.
    # For now, everytime.
    updated_scores = update_eval['perf_score'][model_tag]

    private_key = f"{task}__SDK__{eval_key}"
    private_rank = db_ranking.get(private_key)
    if not private_rank:
        private_rank = dict(Ranking(task_privacy=private_key))
        db_ranking.create(private_key, private_rank)

    new_ranks = []
    # TODO : Sorting on the submit dates in case of problems.
    # Remove if same tag item exists.
    for tag, cur_date, score in private_rank['ranks']:
        if tag != model_tag:
            new_ranks.append((tag, cur_date, score))
    # Add the latest item.
    from collections import deque
    new_ranks = deque(new_ranks)
    new_ranks.appendleft((model_tag, submit_at, updated_scores))
    new_ranks = list(new_ranks)

    # if len(new_ranks) > 10:
    #     new_ranks = new_ranks[-10:]
    private_rank_update = {
        'ranks': new_ranks
    }
    rank_res = db_ranking.update(private_key, private_rank_update)
    print(rank_res)


# TODO : Task queue based
@app.get("/sdk/read/")
async def read(id: Identifier):
    if id not in objects:
        return ServerMessage(action=ServerAction.NOT_INIT)
    obj = objects[id]
    if obj.msg_queue:
        return obj.msg_queue.popleft()
    else:
        return ServerMessage(action=ServerAction.EMPTY)


# TODO : Task queue based
@app.post("/sdk/send/")
async def send(payload: ClientMessage, bg_tasks: BackgroundTasks):
    id = payload.id
    action = payload.action
    data = payload.data
    if action == ClientAction.RUN:
        task = data["task"]
        id = Identifier(task=task, uid=str(uuid.uuid4()))

        bg_tasks.add_task(run_eval_loop, id, task)
        return ServerMessage(action=ServerAction.RUN_ACK, id=id)
    elif action == ClientAction.OUTPUT:
        task_id = data["task_id"]
        idx = data["idx"]
        prompt = data["prompt"].split('\n')[-1]
        res = objects[id].labels[task_id]
        lbl = res[idx]
        pred = data['output']
        if 'output' not in data or not pred:
            is_match = False
        else:
            print(f"LOG : {pred[-5:]} & {lbl}")
            is_match = pred[-1] == lbl[0] or \
                       pred[0] == lbl[0]
        obj = objects[id]
        obj.evals[task_id] += 1
        obj.matches[task_id] += 1 if is_match else 0
        # TODO : Stream send?
        if not is_match:
            obj.samples[task_id].append({
                "Category": task_id,
                "No": idx,
                "Prompt": prompt,
                "Response": pred,
                "Truth": lbl,
                "Correct": "O" if is_match else "X"
            })
        return ServerMessage(action=ServerAction.ACK,
                             data={"idx": idx, "result": is_match})
    elif action == ClientAction.ANALYZE:
        # TODO : Store report to DB
        task = data["task"]

        init_data = TaskPayload(**data)
        eval_key = init_data.eval_key
        upload = init_data.upload
        model_tag = init_data.model_tag

        # TODO : Unify with entry_notifier.py

        obj = objects[id]
        import collections
        res = {}
        sample_res = []
        overall_matches = 0
        overall_evals = 0
        for task_id in TASK_LISTS[task]:
            matches = obj.matches[task_id]
            evals = obj.evals[task_id]
            samples = obj.samples[task_id]

            overall_matches += matches
            overall_evals += evals

            res[task_id] = {
                "matches": matches,
                "evals": evals,
                "accuracy": matches / evals
            }
            sample_res.extend(samples)
        res[f"{TASK_ID[task]}_overall"] = {
            "matches": overall_matches,
            "evals": overall_evals,
            "accuracy": overall_matches / overall_evals
        }
        print(f"LOG : Result for {id}, {task} is {res}")
        print(f"Samples : {sample_res}")

        if upload:
            _upload(res, sample_res, eval_key, model_tag, task)
        return ServerMessage(action=ServerAction.ANALYZE_ACK, data=res)
    elif action == ClientAction.UPLOAD_ONLY:
        res = data["result"]
        task = data["task"]

        init_data = TaskPayload(**data)
        eval_key = init_data.eval_key
        model_tag = init_data.model_tag

        # TODO : Uplaod samples for local
        _upload(res, [], eval_key, model_tag, task)



import uvicorn
from uvicorn.config import LOGGING_CONFIG

if __name__ == '__main__':
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = (
            "%(asctime)s " + LOGGING_CONFIG["formatters"]["access"]["fmt"]
    )
    uvicorn.run('main:app', port=8080, reload=False)
