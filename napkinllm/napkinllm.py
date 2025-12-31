#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2025 by Marek Wydmuch

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import gzip
import json
import pickle
import platform
import multiprocessing
from typing import Any
from math import gcd
from abc import ABC, abstractmethod

from tqdm import tqdm

try:
    import torch
    import torch.multiprocessing as torch_mp
    from torch.utils.data import Dataset
    import transformers

    _LOCAL_AVAILABLE = True
except ImportError:
    _LOCAL_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


try:
    #from vllm import LLM, SamplingParams, GuidedDecodingRequest
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    from vllm.logger import init_logger
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    logger = init_logger(__name__)

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    

# Base LLM Provider
class LLMProvider(ABC):
    def __init__(self, model, task="gen", engine_params=None):
        self.model = model
        self.task = task

    @staticmethod
    def factory(engine, model, task="gen", engine_params=None, mode="local"):
        engine = engine.lower()
        mode = mode.lower()
        if mode == "local":
            if engine == "vllm":
                if not _VLLM_AVAILABLE:
                    raise ValueError("vLLM is not available, please install it using `pip install napkinLLM[vllm]`")
        
                return VLLMProvider(model, task=task, engine_params=engine_params)
            elif engine in ["hf", "huggingface"]:
                return HFProvider(model, task=task, engine_params=engine_params)
            elif engine in ["openai", "claude", "anthropic"]:
                raise ValueError(f"Engine {engine} requires mode='api'")
            else:
                raise ValueError(f"Engine {engine} is not supported")
        elif mode == "api":
            if engine in ["openai", "gpt"]:
                return OpenAIProvider(model, task=task, engine_params=engine_params)
            elif engine in ["claude", "anthropic"]:
                return ClaudeProvider(model, task=task, engine_params=engine_params)
            elif engine in ["vllm", "hf", "huggingface"]:
                raise ValueError(f"Engine {engine} requires mode='local'")
            else:
                raise ValueError(f"Engine {engine} is not supported")
        else:
            raise ValueError(f"Mode {mode} is not supported")
    
    def _check_task(self, method):
        if (self.task == "gen" and method == "embed") or (self.task == "embed" and (method == "generate" or method == "logprobs")):
            raise ValueError(f"{method} is not supported when the task is set to '{self.task}'")

    @abstractmethod
    def embed(self, inputs):
        """
        Returns the embeddings of the inputs as a list of lists of floats
        """
        pass

    @abstractmethod
    def logprobs(self, inputs):
        pass

    @abstractmethod
    def generate(self, inputs, sampling_params=None, guided_options_request=None, apply_chat_template=False, stop_tokens=None):
        pass
    
    
def _ensure_local_dependencies():
    if not _LOCAL_AVAILABLE:
        raise ValueError("Local inference requires torch and transformers. Install with `pip install napkinLLM[local]` or use mode='api'")


def _ensure_openai_available():
    if not _OPENAI_AVAILABLE:
        raise ValueError("OpenAI SDK is not available, please install it using `pip install napkinLLM[openai]`")


def _ensure_anthropic_available():
    if not _ANTHROPIC_AVAILABLE:
        raise ValueError("Anthropic SDK is not available, please install it using `pip install napkinLLM[claude]`")


def _usage_to_dict(usage):
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return dict(usage)


def _normalize_api_messages(item):
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if isinstance(item, dict):
        return [item]
    if isinstance(item, list):
        return item
    raise ValueError("Unsupported input format for API chat, expected string or list of messages")


def _split_system_messages(messages):
    system_parts = []
    filtered_messages = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(str(content))
        else:
            filtered_messages.append(message)
    system = "\n".join([part for part in system_parts if part])
    return system if system else None, filtered_messages


class LocalProvider(LLMProvider):
    def __init__(self, model, task="gen", engine_params=None):
        _ensure_local_dependencies()
        super().__init__(model, task=task, engine_params=engine_params)
        self.model_config = transformers.AutoConfig.from_pretrained(model)
        self.context_len = _dict_get_and_del(engine_params, "context_len", getattr(self.model_config, 'max_position_embeddings', 512))
        self.batch_size = _dict_get_and_del(engine_params, "batch_size", 1)
        self.tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1


class HFProvider(LocalProvider):
    def __init__(self, model, task="gen", engine_params=None):
        super().__init__(model, task=task, engine_params=engine_params)
        
        _model_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": getattr(self.model_config, "attn_implementation", "eager")}
        if "snowflake-arctic-embed" in model:
            _model_kwargs["add_pooling_layer"] = False
        
        _engine_params = dict(
            model_kwargs=_model_kwargs,
            device_map="auto",
            padding=True,
            truncation=True,
            max_length=self.context_len,
        )

        if engine_params is None:
            _engine_params.update(engine_params)

        pipline_task = "text-generation"
        if task == "embed":
            pipline_task = "feature-extraction"

        self.pipeline = transformers.pipeline(
            pipline_task,
            model=model,
            **_engine_params
        )

    @staticmethod
    def _get_dataset(self, inputs):
        class ListDataset(Dataset):
            """Custom dataset for handling list data (e.g. list of texts)"""
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        return ListDataset(inputs),
    

    def embed(self, inputs, embed_params=None):
        self._check_task("embed")
        inputs = HFProvider._get_dataset(inputs)

        embeddings = []
        for batch_inputs in tqdm(inputs):
            batch_embeddings = self.pipeline(batch_inputs, batch_size=self.batch_size, batched=True, normalize=True, pooling='cls')
            print(len(batch_embeddings), type(batch_embeddings))
            embeddings.append(batch_embeddings)

        return embeddings

    def logprobs(self, inputs, sampling_params=None):
        self._check_task("logprobs")
        inputs = HFProvider._get_dataset(inputs)
        raise NotImplementedError("Logprobs task is not implemented for HF engine")

    def generate(self, inputs, sampling_params=None, guided_options_request=None):
        self._check_task("generate")

        # TODO: Apply chat template if instruct model

        inputs = HFProvider._get_dataset(inputs)
    
        if sampling_params is None:
            sampling_params = {}
        return self.pipeline(inputs, **sampling_params)
    

class OpenAIProvider(LLMProvider):
    def __init__(self, model, task="gen", engine_params=None):
        _ensure_openai_available()
        super().__init__(model, task=task, engine_params=engine_params)
        engine_params = engine_params or {}

        api_key = _dict_get_and_del(engine_params, "api_key", os.environ.get("OPENAI_API_KEY"))
        base_url = _dict_get_and_del(
            engine_params,
            "base_url",
            os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        )
        organization = _dict_get_and_del(
            engine_params,
            "organization",
            os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION")
        )
        project = _dict_get_and_del(engine_params, "project", os.environ.get("OPENAI_PROJECT_ID"))
        timeout = _dict_get_and_del(engine_params, "timeout", None)
        max_retries = _dict_get_and_del(engine_params, "max_retries", None)

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        if project:
            client_kwargs["project"] = project
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

        self.client = OpenAI(**client_kwargs)

    def embed(self, inputs, embed_params=None):
        self._check_task("embed")
        if any(not isinstance(item, str) for item in inputs):
            raise ValueError("Embedding inputs must be strings for OpenAI API")

        embed_params = embed_params or {}
        response = self.client.embeddings.create(model=self.model, input=inputs, **embed_params)
        return [item.embedding for item in response.data]

    def logprobs(self, inputs, sampling_params=None):
        self._check_task("logprobs")
        raise NotImplementedError("Logprobs task is not implemented for OpenAI API engine")

    def generate(self, inputs, sampling_params=None, guided_options_request=None, apply_chat_template=False, stop_tokens=None):
        self._check_task("generate")
        sampling_params = sampling_params or {}
        if stop_tokens is not None and "stop" not in sampling_params:
            sampling_params["stop"] = stop_tokens

        outputs = []
        for item in inputs:
            messages = _normalize_api_messages(item)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **sampling_params
            )
            choice = response.choices[0] if response.choices else None
            content = choice.message.content if choice and choice.message else None
            outputs.append({
                "text": content,
                "usage": _usage_to_dict(response.usage),
            })
        return outputs


class ClaudeProvider(LLMProvider):
    def __init__(self, model, task="gen", engine_params=None):
        _ensure_anthropic_available()
        super().__init__(model, task=task, engine_params=engine_params)
        engine_params = engine_params or {}

        api_key = _dict_get_and_del(engine_params, "api_key", os.environ.get("ANTHROPIC_API_KEY"))
        base_url = _dict_get_and_del(engine_params, "base_url", os.environ.get("ANTHROPIC_BASE_URL"))
        timeout = _dict_get_and_del(engine_params, "timeout", None)
        max_retries = _dict_get_and_del(engine_params, "max_retries", None)

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

        self.client = Anthropic(**client_kwargs)

    def embed(self, inputs, embed_params=None):
        self._check_task("embed")
        raise NotImplementedError("Embedding task is not implemented for Claude API engine")

    def logprobs(self, inputs, sampling_params=None):
        self._check_task("logprobs")
        raise NotImplementedError("Logprobs task is not implemented for Claude API engine")

    def generate(self, inputs, sampling_params=None, guided_options_request=None, apply_chat_template=False, stop_tokens=None):
        self._check_task("generate")
        outputs = []

        for item in inputs:
            params = dict(sampling_params or {})
            if stop_tokens is not None and "stop_sequences" not in params:
                params["stop_sequences"] = stop_tokens

            messages = _normalize_api_messages(item)
            system, messages = _split_system_messages(messages)
            if system:
                params["system"] = system
            max_tokens = params.pop("max_tokens", 1024)

            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                **params
            )
            text_parts = []
            for block in response.content:
                if isinstance(block, dict):
                    text = block.get("text")
                else:
                    text = getattr(block, "text", None)
                if text:
                    text_parts.append(text)
            outputs.append({
                "text": "".join(text_parts),
                "usage": _usage_to_dict(response.usage),
            })
        return outputs

if _VLLM_AVAILABLE:
    class VLLMProvider(LocalProvider):
        def __init__(self, model, task="gen", engine_params=None):
            super().__init__(model, task=task, engine_params=engine_params)
            
            _engine_params = dict(
                tensor_parallel_size=gcd(self.tensor_parallel_size, getattr(self.model_config, "num_attention_heads", self.tensor_parallel_size)),
                enable_prefix_caching=True,
                gpu_memory_utilization=0.9,
                dtype="bfloat16",
                trust_remote_code=True,
                #max_model_len=self.context_len,
                max_num_seqs=self.batch_size,
                #tokenizer_config={"model_max_length": _model_len, "truncation": True, "padding": 'max_length'}
            )

            # If it's not a linux machine, then it is cpu only build
            if platform.system() != 'Linux':
                _engine_params.update(dict(
                    tensor_parallel_size=1,
                    enable_prefix_caching=False,
                    dtype="float16",
                    enforce_eager=True,
                ))
            
            if engine_params is not None:
                _engine_params.update(engine_params)

            if task == "embed":
                _engine_params["task"] = "embed"

            if _engine_params["tensor_parallel_size"] > 1:
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

            print(f"{_engine_params=}")

            self.pipeline = LLM(
                model,
                **_engine_params
            )

        def embed(self, inputs, embed_params=None):
            self._check_task("embed")
            print(f"Running LLM embeddings for {len(inputs)} inputs")
            outputs = self.pipeline.embed(inputs, use_tqdm=True)
            outputs = [o.outputs.embedding for o in outputs]
            return outputs

        def logprobs(self, inputs, sampling_params=None):
            self._check_task("logprobs")
            raise NotImplementedError("Logprobs task is not implemented for vLLM engnie")

        def generate(self, inputs, sampling_params=None, guided_options_request=None):
            self._check_task("generate")
            print(f"Running LLM generation for {len(inputs)} inputs")

            # TODO: Apply chat template if instruct model

            # Set params
            if sampling_params is not None and not isinstance(sampling_params, SamplingParams):
                sampling_params = SamplingParams(**sampling_params)

            # TODO: Add nice way to hendle GuidedDecodingOptions
            # if guided_options_request is not None and not isinstance(guided_options_request, GuidedDecodingRequest):
            #     guided_options_request = GuidedDecodingRequest(**guided_options_request)

            return self.pipeline.generate(inputs, sampling_params=sampling_params, guided_options_request=guided_options_request)



# Helpers
INPUT_FORMATS = [".jsonl", ".txt"]
OUTPUT_FORMATS = [".jsonl", ".pkl"]

def _dict_get_and_del(dict_, key, default):
    val = default
    if dict_ and key in dict_:
        val = dict_[key]
        del dict_[key]
    return val

def _filter_params_using_prefix(all_params, prefix):
    """
    """
    _params = {}
    prefix = prefix + "__"
    for k, v in all_params.items():
        if k.startswith(prefix):
            _params[k[len(prefix):]] = v
    return _params

def _split_list(list_, n):
    """
    Splits a list into n roughly equally sized pieces e.g.:
    split_list(["a", "b", "c", "d", "e", "f", "g"], 3) -> [['a', 'b'], ['c', 'd'], ['e', 'f', 'g']]
    """
    return [list_[i * len(list_) // n: (i + 1) * len(list_) // n] for i in range(n)]


def _validate_input_path(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f"Input {file_path} path does not exist")

    if os.path.splitext(file_path)[-1] == ".gz":
        file_path = file_path.replace(".gz", "")
          
    if os.path.splitext(file_path)[-1] not in INPUT_FORMATS:
        raise ValueError(f"Input file format not supported, supported formats are: {', '.join(INPUT_FORMATS)}")


def _read_input(file_path, input_range=None):
    file = None
    file_ext = os.path.splitext(file_path)[-1]
    if file_ext == ".gz":
        file_ext = os.path.splitext(file_path.replace(".gz", ""))[-1]
        file = gzip.open(file_path, "rt")
    else:
        file = open(file_path, "r")
    
    input_lines = file.readlines()

    # Possible optimization: read only a range of lines
    if input_range is not None:
        input_lines = input_lines[input_range[0]:input_range[1]]

    if file_ext == ".jsonl":
        input = [json.loads(line) for line in input_lines]
    
    file.close()
    return input


def _validate_output_path(file_path):
    if os.path.splitext(file_path)[-1] == ".gz":
        file_path = file_path.replace(".gz", "")

    if os.path.splitext(file_path)[-1] not in OUTPUT_FORMATS:
        raise ValueError(f"Output file format not supported, supported formats are: {', '.join(OUTPUT_FORMATS)}")


def _write_output(file_path, outputs):
    file = None
    file_ext = os.path.splitext(file_path)[-1]
    if file_ext == ".pkl":
        file = open(file_path, "wb")
        pickle.dump(outputs, file)
        file.close()
        return
    
    elif file_ext == ".gz":
        file_ext = os.path.splitext(file_path.replace(".gz", ""))[-1]
        file = gzip.open(file_path, "wt")
    else:
        file = open(file_path, "w")
    
    if isinstance(outputs, list):
        for output in outputs:
            file.write(json.dumps(output) + "\n")
    else:
        for output in outputs:
            file.write(output + "\n")
    file.close()


def _run_inference_subprocess(
    task: str,
    model: str,
    inputs: list[str] | list[list[dict[str, str]]],
    mode: str,
    engine: str,
    gpus: int | str | list[int] | None,
    kwargs_dict: dict[str, Any],
    ):
    # Set gpu visibility
    if gpus is not None:
        if isinstance(gpus, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in gpus])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"Running subprocess {multiprocessing.current_process()} with {gpus=}, CUDA_VISIBLE_DEVICES={cuda_visible}")
    return run_inference(task, model, inputs, mode=mode, engine=engine, **kwargs_dict)

def run_inference(    
    task: str,
    model: str,
    inputs: list[str] | list[list[dict[str, str]]],
    mode: str = "local",
    engine: str = "vllm",
    **kwargs):

    # Set engine params
    engine_params = _filter_params_using_prefix(kwargs, "engine")   
    llm = LLMProvider.factory(engine, model, task=task, engine_params=engine_params, mode=mode)

    # Run task
    if task == "gen":
        sampling_params = _filter_params_using_prefix(kwargs, "sampling")
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    elif task == "logprobs":
        sampling_params = _filter_params_using_prefix(kwargs, "sampling")
        outputs = llm.logprobs(inputs, sampling_params=sampling_params)

    elif task == "embed":
        embed_params = _filter_params_using_prefix(kwargs, "embedding")
        outputs = llm.embed(inputs, embed_params=embed_params)

    return outputs

# Main CLI function
TASK_EMOJIS = {
    "gen": "🖊️",
    "logprobs": "🔍",
    "embed": "🤿",
}

MODEL_EMOJIS = {
    "llama": "🦙",
    "arctic": "❄️",
    "mistral": "🌪️",
    "bert": "🐦",
    "qwen": "🌊",
    "deepseek": "",
    "claude": "🎼",
    "gpt": "🤖"

    # Not supported/tested yet
    # "gemini": "♊",
    # "palm": "🌴",
    # "phi": "φ",
    # "bloom": "🌸",
    # "falcon": "🦅",
}

ENGINE_EMOJIS = {
    "hf": "🤗",
    "huggingface": "🤗",
    "vllm": "🚀",
    "openai": "🤖",
    "gpt": "🤖",
    "claude": "🎼",
    "anthropic": "🎼",
}

def _find_emoji(text, emojis, default=""):
    text = text.lower()
    for k, e in emojis.items():
        if k in text:
            return e
    return default

def napkinllm(
    task: str,
    model: str,
    input_path: str,
    output_path: str,
    mode: str = "local",
    engine: str = "vllm",
    parallel: int = 1,
    input_range: tuple[int, int] | None = None,
    **kwargs
):
    engine_key = engine.lower()
    header = f"""
 |`\                            _     _         _      _      __  __ 
 |  `\     _ __    __ _  _ __  | | __(_) _ __  | |    | |    |  \/  |
 |    )   | '_ \  / _` || '_ \ | |/ /| || '_ \ | |    | |    | |\/| |
 |  ,//   | | | || (_| || |_) ||   < | || | | || |___ | |___ | |  | |
 |,/ /    |_| |_| \__,_|| .__/ |_|\_\|_||_| |_||_____||_____||_|  |_|
   \/                   |_|                                          

{TASK_EMOJIS[task]} Task: {task}
{_find_emoji(model, MODEL_EMOJIS, default=["💃","🕺"])} Model: {model}
⬅️ Input: {input_path}
➡️ Output: {output_path}
{ENGINE_EMOJIS.get(engine_key, "")} Engine: {engine}
🏷️ Mode: {mode}
🎛️ Args: {kwargs}
"""
    print(header)

    # Validate input and output paths
    _validate_input_path(input_path)
    _validate_output_path(output_path)

    # Read input
    inputs = _read_input(input_path, input_range=input_range)
    print(f"Read {len(inputs)} inputs")

    if parallel > 1:
        mode = mode.lower()
        if mode == "local":
            if not _LOCAL_AVAILABLE:
                raise ValueError("Local inference requires torch and transformers. Install with `pip install napkinLLM[local]` or use mode='api'")
            if not torch.cuda.is_available() or torch.cuda.device_count() < parallel:
                raise ValueError(f"Cannot run inference in {parallel} processes without at least {parallel} GPUs")

            # Set start method to spawn
            torch_mp.set_start_method('spawn')
            if engine == "vllm":
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

            gpus = list(range(torch.cuda.device_count()))
            splited_inputs = _split_list(inputs, parallel)
            splited_gpu_ids = _split_list(gpus, parallel)
            subprocess_args = [(task, model, i, mode, engine, g, kwargs) for i, g in zip(splited_inputs, splited_gpu_ids)]
            print(f"Running inference in parallel in {parallel} processes, on {splited_gpu_ids} gpus")

            #with multiprocessing.Pool(processes=parallel) as pool:
            with torch_mp.Pool() as pool:
                pool_outputs = pool.starmap(_run_inference_subprocess, subprocess_args)
        elif mode == "api":
            mp = multiprocessing
            mp.set_start_method('spawn')
            splited_inputs = _split_list(inputs, parallel)
            subprocess_args = [(task, model, i, mode, engine, None, kwargs) for i in splited_inputs]
            print(f"Running inference in parallel in {parallel} processes with API engine")

            with mp.Pool() as pool:
                pool_outputs = pool.starmap(_run_inference_subprocess, subprocess_args)
        else:
            raise ValueError(f"Mode {mode} is not supported")

        outputs = []
        for o in pool_outputs:
            outputs.extend(o)
    else:
        outputs = run_inference(task, model, inputs, mode=mode, engine=engine, **kwargs)

    # Write output
    _write_output(output_path, outputs)


if __name__ == "__main__":
    from fire import Fire

    Fire(napkinllm)
