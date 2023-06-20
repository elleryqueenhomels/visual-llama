import copy
import json
import random

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions

from llama import Tokenizer
from llama import format_prompt


def construct_sample(ann: dict, tokenizer: Tokenizer, max_words: int):
    prompt = format_prompt(ann["instruction"], ann.get("input", ""))
    example = prompt + ann["output"]
    prompt = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
    example = torch.tensor(tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
    padding = max_words - example.shape[0]
    if padding > 0:
        example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        example = example[: max_words]
    labels = copy.deepcopy(example)
    labels[: len(prompt)] = -1
    example_mask = example.ge(0)
    label_mask = labels.ge(0)
    example[~example_mask] = 0
    labels[~label_mask] = 0
    example_mask = example_mask.float()
    label_mask = label_mask.float()
    return example, labels, example_mask


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_words=30, partition="train"):
        self.ann = json.load(open(data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        return construct_sample(self.ann[index], self.tokenizer, self.max_words)


class JointDataset(Dataset):
    def __init__(self, args, img_transform_fn, tokenizer: Tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.img_transform_fn = img_transform_fn
        self.cap_dataset = CocoCaptions(root=args.coco_imgs_dir, annFile=args.coco_ann_file)
        self.inst_dataset = InstructionDataset(data_path=args.inst_path, tokenizer=tokenizer, max_words=args.max_seq_len)
    
    def __len__(self):
        return len(self.cap_dataset) + len(self.inst_dataset)
    
    def __getitem__(self, index):
        imgs = None
        tokens = None
        labels = None
        if index % 2 == 0:
            tokens, labels, _ = self.inst_dataset[index // 2]
            imgs = torch.zeros(3, 224, 224)
        else:
            imgs, targets = self.cap_dataset[index // 2]
            imgs = self.img_transform_fn(imgs)
            target = targets[random.randint(0, len(targets) - 1)]
            ann = {"instruction": "Describe the image", "output": target}
            tokens, labels, _ = construct_sample(ann, self.tokenizer, self.args.max_seq_len)
        imgs = imgs.to(tokens.device)
        return tokens, imgs, labels
