from . import gpt2
from . import gpt3
from . import azure_gpt
from . import anthropic_llms
from . import huggingface
from . import textsynth
from . import dummy
from . import chatgpt
from . import custom

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "hf-causal": gpt2.HFLM,
    "hf-causal-experimental": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "azure": azure_gpt.AzureChatGPTLM,
    "custom": custom.CustomLM,
    "anthropic": anthropic_llms.AnthropicLM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
    'chatgpt': chatgpt.ChatGPT
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
