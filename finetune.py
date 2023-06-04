import argparse
import os
import json
import random
import numpy as np

from pathlib import Path
from huggingface_hub import hf_hub_download
from llama import ModelArgs, VisionModel, Tokenizer, Transformer

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions

import timm.optim.optim_factory as optim_factory
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from fairscale.nn.model_parallel.initialize import initialize_model_parallel


def setup_model_parallel():
    init_process_group(backend="nccl")
    initialize_model_parallel(int(os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def setup_random_seeds(random_seed=0):
    # fix the seed for reproducibility
    torch.manual_seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


class Trainer:
    def __init__(
        self,
        args,
        tokenizer: Tokenizer,
        model: Transformer,
        vision_model: VisionModel,
        train_data: DataLoader,
    ) -> None:
        self.args = args
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.vision_model = vision_model
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        param_groups = optim_factory.add_weight_decay(self.model.module, args.weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        self.epochs_run = 0
        self.snapshot_path = args.resume
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()

    def _load_snapshot(self):
        snapshot = torch.load(self.snapshot_path, map_location='cpu')
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_batch(self, tokens, visual_tokens, labels):
        self.optimizer.zero_grad()
        loss = self.model.forward_train(tokens, visual_tokens, labels)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for (tokens, imgs), labels in self.train_data:
            tokens = tokens.to(self.gpu_id)
            visual_tokens = self.vision_model(imgs) if imgs is not None else None
            self._run_batch(tokens, visual_tokens, labels)

    def train(self):
        for epoch in range(self.epochs_run, self.args.epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.args.save_every == 0:
                self._save_snapshot(epoch)


class InstructionDataset(Dataset):
    def __init__(self):
        pass


class JointDataset(Dataset):
    def __init__(self):
        pass


def build_model(args):
    param_path = hf_hub_download(
        repo_id="nyanko7/LLaMA-7B", filename="params.json")
    tokenizer_path = hf_hub_download(
        repo_id="nyanko7/LLaMA-7B", filename="tokenizer.model")
    model_path = hf_hub_download(
        repo_id="nyanko7/LLaMA-7B", filename="consolidated.00.pth")

    with open(param_path, "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=args.max_seq_len, max_batch_size=args.batch_size, **params)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    gpu_id = int(os.environ["LOCAL_RANK"])

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    vision_model = VisionModel(model_args).cuda(gpu_id)
    model = Transformer(model_args).cuda(gpu_id)

    # To reduce memory usuage
    model_ckpt = torch.load(model_path, map_location=f'cuda:{gpu_id}')
    model.load_state_dict(model_ckpt, strict=False)
    del model_ckpt
    torch.cuda.empty_cache()

    for name, param in model.named_parameters():
        requires_grad = (
            name == "adapter_query"
            or name.endswith(".gate")
            or name.endswith("_bias")
            or name.endswith("_scale")
            or "lora_w" in name
        )
        if requires_grad:
            param.data = param.data.float()
            param.requires_grad = True
        else:
            param.requires_grad = False

    return tokenizer, model, vision_model


def load_dataset():
    train_set = JointDataset()
    return train_set


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset))


def main(args):
    setup_model_parallel()
    setup_random_seeds(args.seed)
    dataset = load_dataset()
    train_data = prepare_dataloader(dataset, args.batch_size)
    tokenizer, model, vision_model = build_model(args)
    trainer = Trainer(args, tokenizer, model, vision_model, train_data)
    trainer.train()
    destroy_process_group()


def get_args_parser():
    parser = argparse.ArgumentParser("LLaMA fine-tuning", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument("--llama_model_path", default="./llama", type=str, help="path of llama model")
    parser.add_argument("--model", default="llama7B_adapter", type=str, metavar="MODEL", help="Name of model to train")

    parser.add_argument("--adapter_layer", type=int, default=30, metavar="LENGTH", help="the number of adapter layer")

    parser.add_argument("--adapter_len", type=int, default=10, metavar="LENGTH", help="the adapter length")

    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH", help="the maximum sequence length")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--data_path", default="/instruction_dataset/", type=str, help="dataset path")

    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument('--save_every', default=8, type=int, help='How often to save a snapshot')

    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)