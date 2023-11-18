#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
KOBEST
https://arxiv.org/abs/2204.04541

A well-formulated benchmark plays a critical role in spurring advancements 
in the natural language processing (NLP) field, as it allows objective and
precise evaluation of diverse models. As modern language models (LMs) have 
become more elaborate and sophisticated, more difficult benchmarks that require
linguistic knowledge and reasoning have been proposed. However, most of these
benchmarks only support English, and great effort is necessary to construct
benchmarks for other low resource languages. To this end, we propose a new
benchmark named Korean balanced evaluation of significant tasks (KoBEST),
which consists of five Korean-language downstream tasks. Professional Korean
linguists designed the tasks that require advanced Korean linguistic knowledge.
Moreover, our data is purely annotated by humans and thoroughly reviewed to
guarantee high data quality. We also provide baseline models and human performance
results. Our dataset is available on the Huggingface.

Homepage: https://huggingface.co/datasets/skt/kobest_v1
"""

import numpy as np
from lm_eval.base import MultipleChoiceTask, rf, Task
from lm_eval.metrics import macro_f1_score, mean


class BoolQ(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "skt/kobest_v1"
    DATASET_NAME = "boolq"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return doc['query']
    
    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {
            "acc": pred == gold,
            "macro_f1": (gold, pred)
        }
        
    def _process_doc(self, doc):
        out_doc = {
            "query": "{} 질문: {} 답변: ".format(doc["paragraph"], doc["question"]),
            "choices": ["아니요", "예"],
            "gold": int(doc['label']),
        }
        return out_doc

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }


class COPA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "skt/kobest_v1"
    DATASET_NAME = "copa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return doc['query']

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {
            "acc": pred == gold,
            "macro_f1": (gold, pred)
        }
    
    def _process_doc(self, doc):
        connector = {
            "원인": "왜냐하면",
            "결과": "그래서",
        }[doc["question"].strip()]
        
        out_doc = {
            "query": doc["premise"] + f" {connector}",
            "choices": [doc["alternative_1"], doc["alternative_2"]],
            "gold": int(doc['label']),
        }
        return out_doc

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }

class WiC(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "skt/kobest_v1"
    DATASET_NAME = "wic"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])
    
    def doc_to_text(self, doc):
        return doc['query']

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {
            "acc": pred == gold,
            "macro_f1": (gold, pred)
        }
        
    def _process_doc(self, doc):
        out_doc = {
            "query": "문장1: {} 문장2: {} 두 문장에서 {}가 같은 뜻으로 쓰였나?".format(doc["context_1"], doc["context_2"], doc["word"]),
            "choices": ["아니요", "예"],
            "gold": int(doc['label']),
        }
        return out_doc

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }


class HellaSwag(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "skt/kobest_v1"
    DATASET_NAME = "hellaswag"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "query": "문장: {}".format(doc["context"]),
            "choices": [doc["ending_1"], doc["ending_2"], doc["ending_3"], doc["ending_4"]],
            "gold": int(doc['label']),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]
    
    def construct_ko_text(self, doc, ctx):
        answer_keys = self.answer_candidates(doc)
        prompt = ctx + '\n정답후보:'
        for i, key in enumerate(answer_keys):
            text = f" {key} {doc['choices'][i]}"
            prompt += text
        prompt += "\n정답:"
        
        return prompt

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["gold"]
        return {
            "acc": pred == gold,
            "macro_f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }


class SentiNeg(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "skt/kobest_v1"
    DATASET_NAME = "sentineg"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return doc['query']
    
    def _process_doc(self, doc):
        out_doc = {
            "query": "문장: {} 긍부정:".format(doc["sentence"]),
            "choices": ["부정", "긍정"],
            "gold": int(doc['label']),
        }
        return out_doc

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {
            "acc": pred == gold,
            "macro_f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }
