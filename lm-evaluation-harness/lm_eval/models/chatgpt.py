#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import transformers
from lm_eval.base import BaseLM
from lm_eval import utils
from tqdm import tqdm
import time


def oa_completion(**kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    import openai

    backoff_time = 3
    while True:
        try:
            return openai.ChatCompletion.create(**kwargs)
        except openai.error.OpenAIError:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5


class ChatGPT(BaseLM):
    REQ_CHUNK_SIZE = 20

    def __init__(self, engine='chatgpt', truncate=False, token_length_norm=True):
        """

        :param engine: str
            OpenAI API engine (e.g. davinci)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        self.token_length_norm = token_length_norm

        import openai
        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = 'fabf3a06cfac4534969573afa0f4fdb8'
        openai.api_base = 'https://mono.openai.azure.com/'
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15'
        # openai.api_key = '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'
        self.engine = engine

        # to make the annoying "Using pad_token, but it is not set yet." error go away
        self.truncate = truncate

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def get_result(self, prompt):
        inp = [{'role': 'user', 'content': prompt}]
        response = oa_completion(
            engine=self.engine,
            # model=self.engine,
            messages=inp,
            max_tokens=self.max_gen_toks,
            temperature=0.0,
        )
    
        return response

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
