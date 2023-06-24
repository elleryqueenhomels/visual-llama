import argparse
import os
import sys
import math
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import timm.optim.optim_factory as optim_factory
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

import misc
from llama import VisionModel, Tokenizer, Transformer
from llama import load_model
from dataset import JointDataset


class Trainer:
    def __init__(
        self,
        args,
        tokenizer: Tokenizer,
        model: Transformer,
        vision_model: VisionModel,
        train_data: DataLoader,
    ):
        self.args = args
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.train_data = train_data
        self.tokenizer = tokenizer
        self.vision_model = vision_model

        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id], static_graph=True)

        param_groups = optim_factory.add_weight_decay(self.model.module, args.weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        self.loss_scaler = misc.NativeScalerWithGradNormCount()

        self.epochs_run = 0
        self.snapshot_path = args.resume
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()

    def _load_snapshot(self):
        snapshot = torch.load(self.snapshot_path, map_location='cpu')
        self.model.load_state_dict(snapshot["model"], strict=False)
        self.vision_model.load_state_dict(snapshot["vision_model"], strict=False)
        self.optimizer.load_state_dict(snapshot["optimizer"], strict=False)
        self.loss_scaler.load_state_dict(snapshot["scaler"], state_dict=False)
        self.epochs_run = snapshot["epochs_run"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "model": self.model.state_dict(),
            "vision_model": self.vision_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.loss_scaler.state_dict(),
            "epochs_run": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_batch(self, tokens, visual_tokens, labels, step):
        with torch.cuda.amp.autocast():
            loss = self.model(tokens, visual_tokens, labels)
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}. Stop training. Exit...")
            sys.exit(1)
        loss /= self.args.accum_iter
        update_grad = (step + 1) % self.args.accum_iter == 0
        self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(), retain_graph=True, update_grad=update_grad)
        if update_grad:
            self.optimizer.zero_grad()
        torch.cuda.synchronize()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU {self.gpu_id}] Epoch: {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

        step = 0
        self.train_data.sampler.set_epoch(epoch)
        self.optimizer.zero_grad()

        for tokens, imgs, labels in self.train_data:
            print(f"Epoch: {epoch} | Step: {step}")

            if step % self.args.accum_iter == 0:
                misc.adjust_learning_rate(self.optimizer, step + len(self.train_data) + epoch, self.args)

            tokens = tokens.to(self.gpu_id)
            visual_tokens = self.vision_model(imgs)
            self._run_batch(tokens, visual_tokens, labels, step)
            step += 1

    def train(self):
        print("Trainable Params of LLaMA:")
        print([(key, val.shape) for key, val in self.model.named_parameters() if val.requires_grad])
        print("Trainable Params of Vision Model:")
        print([(key, val.shape) for key, val in self.vision_model.named_parameters() if val.requires_grad])

        # lr preparation
        eff_batch_size = self.args.batch_size * self.args.accum_iter * self.world_size

        if self.args.lr is None:
            # only base_lr is specified
            self.args.lr = self.args.blr * eff_batch_size / 256

        print("base lr: %.2e" % self.args.blr)
        print("actual lr: %.2e" % self.args.lr)
        print("effective batch size: %d" % eff_batch_size)
        print("accumulate grad iterations: %d" % self.args.accum_iter)

        for epoch in range(self.epochs_run, self.args.epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.args.save_every == 0:
                self._save_snapshot(epoch)


def setup_model_parallel():
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
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


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=False))


def main(args):
    setup_model_parallel()
    setup_random_seeds(args.seed)
    tokenizer, model, vision_model = load_model(
        name="",
        llama_dir=args.llama_model_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        gpu_id=int(os.environ["LOCAL_RANK"]),
        is_training=True,
    )
    dataset = JointDataset(args, vision_model.clip_transform, tokenizer)
    train_data = prepare_dataloader(dataset, args.batch_size)
    trainer = Trainer(args, tokenizer, model, vision_model, train_data)
    trainer.train()
    destroy_process_group()


def get_args_parser():
    parser = argparse.ArgumentParser("LLaMA fine-tuning", add_help=False)

    # Model parameters
    parser.add_argument("--llama_model_path", default="./LLaMA_7B", type=str, help="path of pre-trained llama model")
    parser.add_argument("--adapter_model_path", default="./ckpt", type=str, help="path of trained adapter params")

    parser.add_argument("--max_seq_len", type=int, default=512, help="the maximum sequence length")
    parser.add_argument("--adapter_len", type=int, default=10, help="the adapter length")
    parser.add_argument("--adapter_layer", type=int, default=30, help="the number of adapter layer")

    # Training parameters
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--epochs", default=400, type=int)

    parser.add_argument("--accum_iter", default=1, type=int, help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument("--warmup_epochs", type=int, default=40, help="epochs to warmup LR")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=1e-3, help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument("--min_lr", type=float, default=0.0, help="lower lr bound for cyclic schedulers that hit 0")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")

    # Dataset parameters
    parser.add_argument("--inst_path", default="./instruction_dataset/data.json", type=str, help="instruction dataset path")
    parser.add_argument("--coco_imgs_dir", default="./coco/train2017", type=str, help="coco images dir")
    parser.add_argument("--coco_ann_file", default="./coco/annotations/captions_train2017.json", type=str, help="coco captions json file")

    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")

    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--seed", default=0, type=int, help="fix the seed for reproducibility")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)

    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument('--save_every', default=8, type=int, help='How often to save a snapshot')
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
