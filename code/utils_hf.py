# Author   : Xuanli He
# Version  : 1.0
# Filename : utils.py
# Modified and extended by: Ukyo Honda
from typing import Tuple, Any
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_libs = {
    "phi-4": "unsloth/phi-4",
    "deepseek-r1-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

data_libs = {
    "HANS": "HANS",
    "NAN": "NAN",
    "ISCS": "ISCS",
    "ST": "ISCS",
    "LILI": "ISCS",
    "PICD": "ISCS",
    "PISP": "ISCS",
    "ESNLI": "ESNLI",
    "ANLI": "ANLI",
    "QQP": "QQP",
    "PAWS": "QQP",
}


def create_model(
    model_name: str,
    device: str,
    do_compile: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    quantize: str = "full",
) -> Tuple[Any, Any]:
    model_kwargs = {}

    if device == "cuda":
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["device_map"] = "balanced_low_0"  # 'auto'
    else:
        model_kwargs["low_cpu_mem_usage"] = True

    if quantize == "int4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif quantize == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, resume_download=True, trust_remote_code=True, **model_kwargs
    )

    if do_compile is True:
        model = torch.compile(model)

    model.eval()

    return tokenizer, model


def args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, choices=list(model_libs.keys()))

    parser.add_argument(
        "--train_file",
        type=str,
        help="In-context demonstration examples",
        required=True,
    )

    parser.add_argument(
        "--test_file",
        type=str,
        help="test file for the evaluation",
        required=True,
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=list(data_libs.keys()),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="random seed",
    )

    parser.add_argument("--temp", type=float, default=0.0, help="temperature")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--sample", type=int, default=1, help="number of samples to generate")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--quantize", type=str, default="full", choices=["full", "int4", "int8"])

    return parser
