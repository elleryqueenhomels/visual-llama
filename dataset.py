import os
import copy
import json
import yaml
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions

from llama import Tokenizer
from llama import format_prompt


def construct_sample(instruction: str, input: str, response: str, tokenizer: Tokenizer, max_words: int):
    prompt = format_prompt(instruction, input)
    example = prompt + response
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
        data_item = self.ann[index]
        return construct_sample(
            data_item['instruction'],
            data_item['input'],
            data_item['output'],
            self.tokenizer,
            self.max_words)


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
        image = None
        tokens = None
        labels = None
        if index % 2 == 0:
            tokens, labels, _ = self.inst_dataset[index // 2]
            image = torch.zeros(3, 224, 224)
        else:
            image, targets = self.cap_dataset[index // 2]
            image = self.img_transform_fn(image)
            target = targets[random.randint(0, len(targets) - 1)]
            ann = {"instruction": "Describe the image", "output": target, "input": ""}
            tokens, labels, _ = construct_sample(
                ann['instruction'],
                ann['input'],
                ann['output'],
                self.tokenizer,
                self.args.max_seq_len)
        return tokens, image, labels


class FinetuneDataset(Dataset):
    def __init__(self, config_path, coco_dir, img_transform_fn, tokenizer: Tokenizer, max_words=30):
        print(f"Reading dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)

        ann = []
        for meta_path in self.config['META']:
            data = json.load(open(meta_path))
            print(f"{meta_path}: {len(data)} records")
            ann += data
        self.ann = ann
        print(f"total length: {len(self)}")

        self.img_transform_fn = img_transform_fn
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.coco_dir = coco_dir

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]
        if 'image' in data_item:
            filename = data_item['image']
            question = data_item['conversations'][0]['value']
            answer = data_item['conversations'][1]['value']

            img_path = os.path.join(self.coco_dir, filename)
            image = Image.open(img_path).convert('RGB')
            image = self.img_transform_fn(image)
            format_instruction = question
            format_input = None
        else:
            image = torch.zeros(3, 224, 224)
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']

        tokens, labels, _ = construct_sample(
            format_instruction,
            format_input,
            answer,
            self.tokenizer,
            self.max_words)
        return tokens, image, labels
