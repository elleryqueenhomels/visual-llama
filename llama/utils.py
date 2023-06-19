import os
import json
import urllib
import hashlib
import warnings
from pathlib import Path

from tqdm import tqdm
import torch

from .model import ModelArgs, Transformer, VisionModel
from .tokenizer import Tokenizer


_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def format_prompt(instruction, input=None):
    if input is None:
        return _PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return _PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


_MODELS = {
    "BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    # assume the url is https://some/path/sha256_model.pth
    expected_sha256 = url.split("/")[-1].split('_')[0]
    # expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def load_model(
    name,
    llama_dir,
    max_seq_len=512,
    max_batch_size=1,
    gpu_id=0,
    download_root='ckpts',
):
    adapter_path = None
    if name in _MODELS:
        adapter_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        adapter_path = name
    elif name is not None and name != "":
        return RuntimeError(f"Model {name} not found; available models = {list(_MODELS.keys())}")

    # load model args
    with open(os.path.join(llama_dir, 'params.json'), "r") as f:
        model_args = json.loads(f.read())

    # load adapter weights and model_cfg
    print(f'Loading Adapter from {adapter_path}')
    adapter_cfg = {}
    adapter_ckpt = None
    if adapter_path is not None:
        adapter_ckpt = torch.load(adapter_path, map_location='cpu')
        adapter_cfg = adapter_ckpt.get('config', {})
    model_args = {**model_args, **adapter_cfg}
    model_args = ModelArgs(**model_args)

    # load tokenizer
    tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
    tokenizer = Tokenizer(model_path=tokenzier_path)
    model_args.vocab_size = tokenizer.n_words
    model_args.max_seq_len = max_seq_len
    model_args.max_batch_size = max_batch_size

    # load model
    vision_model = VisionModel(model_args).cuda(gpu_id)
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda(gpu_id)
    torch.set_default_tensor_type(torch.FloatTensor)

    # load weights from checkpoints
    ckpts = sorted(Path(llama_dir).glob("*.pth"))
    for ckpt in ckpts:
        ckpt = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
    if adapter_ckpt is not None:
        model.load_state_dict(adapter_ckpt['model'], strict=False)

    return tokenizer, model, vision_model
