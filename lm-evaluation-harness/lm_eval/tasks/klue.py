"""
KLUE
https://arxiv.org/abs/2105.09680

 Korean Language Understanding Evaluation (KLUE) benchmark is a series of datasets
 to evaluate natural language understanding capability of Korean language models.
 KLUE consists of 8 diverse and representative tasks, which are accessible to anyone without any restrictions.
 With ethical considerations in mind, we deliberately design annotation guidelines
 to obtain unambiguous annotations for all datasets. Furthermore, we build an evaluation system
 and carefully choose evaluations metrics for every task, thus establishing fair comparison across Korean language models.

 Homepage: https://klue-benchmark.com/
"""

import datasets
import numpy as np
from functools import partial
from math import exp
from lm_eval.base import Task, MultipleChoiceTask, rf
from lm_eval.metrics import (
    macro_f1_score,
    mean,
    matthews_corrcoef,
    f1_score,
    yesno,
    micro_f1_score,
)
from lm_eval.utils import general_detokenize

_CITATION = """
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class NLI(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "klue"
    DATASET_NAME = "nli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def doc_to_text(self, doc):
        return doc['query']
        
    def _process_doc(self, doc):
        out_doc = {"query": "{}\n질문: {} 동일, 반대 또는 중립?\n답변:".format(
                    doc["premise"],
                    doc["hypothesis"].strip()
                    + ("" if doc["hypothesis"].strip().endswith(".") else "."),),
            "choices": ["동일", "중립", "반대"],
            "gold": doc["label"],
        }
        return out_doc

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class STS(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "klue"
    DATASET_NAME = "sts"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def doc_to_text(self, doc):
        return doc['query']

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["labels"]["binary-label"]
        return {"acc": pred == gold, "f1": (gold, pred)}
    
    def _process_doc(self, doc):
        out_doc = {
            "query": "질문: 문장 1과 문장 2는 서로 유사한 의미를 가지나요?\n문장 1:{}\n문장 2:{}\n정답:".format(
                        general_detokenize(doc["sentence1"]), general_detokenize(doc["sentence2"])),
            "choices": ["아니요", "예"],
            "gold": doc["labels"]["binary-label"],
        }
        return out_doc

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}


class YNAT(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "klue"
    DATASET_NAME = "ynat"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "query": "제목: {}\n주제:".format(doc["title"]),
            "choices": ["과학", "경제", "사회", "생활", "세계", "스포츠", "정치"],
            "gold": doc["label"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return "{}".format(doc["query"])

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["gold"]
        return {"f1": (gold, pred)}

    def higher_is_better(self):
        return {"f1": True}

    def aggregation(self):
        return {"f1": macro_f1_score}