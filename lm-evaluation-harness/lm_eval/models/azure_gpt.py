#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from lm_eval import utils
from tqdm import tqdm
import time

from lm_eval.base import BaseLM


def get_result(response, ctxlen):
    # Will not work with ChatGPT

    """Process results from Azure OpenAI API response.

    :param response: dict
        Azure OpenAI API Response
    :param ctxlen: int
        Length of context (so we can slice them away and only keep the predictions)
    :return:
        continuation_logprobs: np.array
            Log probabilities of continuation tokens
        is_greedy: bool
            whether argmax matches given continuation exactly
    """
    is_greedy = True
    logprobs = response["logprobs"]["token_logprobs"]
    continuation_logprobs = sum(logprobs[ctxlen:])

    for i in range(ctxlen, len(response["logprobs"]["tokens"])):
        token = response["logprobs"]["tokens"][i]
        top_tokens = response["logprobs"]["top_logprobs"][i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


def oa_completion(**kwargs):
    """Query Azure OpenAI API for completion.

    Retry with back-off until they respond
    """
    import openai

    backoff_time = 3
    while True:
        try:
            return openai.Completion.create(**kwargs)
        except openai.error.OpenAIError:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5

class AzureChatGPTLM(BaseLM):
    REQ_CHUNK_SIZE = 20

    def __init__(self):
        """

        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()

        import openai
        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = os.environ["AZURE_OPENAI_KEY"]
        openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15'

        self.deployment_name = os.environ["AZURE_DEPLOYMENT_NAME"]

    @property
    def eot_token_id(self):
        raise NotImplementedError("No idea about chatgpt tokenization.")

    @property
    def max_length(self):
        # Note: the Azure OpenAI API supports up to 2049 tokens, with the first token being the first input token
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
        raise NotImplementedError("No idea about chatgpt tokenization.")

    def tok_decode(self, tokens):
        raise NotImplementedError("No idea about chatgpt tokenization.")

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        raise NotImplementedError("No support for logits.")

    def greedy_until(self, requests):
        if not requests:
            return []

        res = []
        for request in tqdm(requests):
            inp = request[0]
            request_args = request[1]
            until = request_args["until"]
            response = oa_completion(
                engine=self.deployment_name,
                prompt=inp,
                max_tokens=self.max_gen_toks,
                temperature=0.0,
                stop=until,
            )
            res.append(response["text"])
        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()