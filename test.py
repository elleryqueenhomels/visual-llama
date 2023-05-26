import os
import sys
import torch

from PIL import Image
from huggingface_hub import hf_hub_download
from torch.distributed import is_initialized
from llama import LLaMA, ModelArgs, Tokenizer, Transformer, VisionModel
from fairscale.nn.model_parallel.initialize import initialize_model_parallel


def setup_model_parallel():
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MP'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2223'
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    if not is_initialized():
        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

local_rank, world_size = setup_model_parallel()
if local_rank > 0:
    sys.stdout = open(os.devnull, "w")


model_args = ModelArgs(
    dim = 64,
    n_layers = 2,
    n_heads = 2,
    vocab_size = 32,
    multiple_of = 32,
    max_seq_len = 512,
    adapter_len = 10,
    adapter_layer = 1,
    vision_clip_model = "ViT-L/14",
    vision_dim = 64,
    vision_blocks = 2,
    vision_early_fusion = {0},
    add_bias=True,
    add_scale=True,
    use_lora=True,
)

tokenizer_path = hf_hub_download(
    repo_id="huggyllama/llama-7b", filename="tokenizer.model")
tokenizer = Tokenizer(model_path=tokenizer_path)
model_args.vocab_size = tokenizer.n_words

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = Transformer(model_args).cuda()
vision_model = VisionModel(model_args).cuda()

torch.set_default_tensor_type(torch.FloatTensor)
generator = LLaMA(model, tokenizer, vision_model)

img = '/gdrive/MyDrive/ML Colab Sessions/LLaMA Playground/Lance.jpg'
imgs = [Image.open(img).convert('RGB')]
output = generator.generate(['Tell me a story'], imgs=imgs, temperature=0.0)
print(output)
