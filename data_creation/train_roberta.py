import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import json
import io
import os

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

PROMPT_DICT = {
    "prompt_input": (
        "Input:\n{input}"
    ),
    "prompt_no_input": (
        "{instruction}"
    ),
}


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu()
                          for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer):
        super(SupervisedDataset, self).__init__()

        data = jload('../gpt4_reward_all_0813_train.json')

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT[
            "prompt_no_input"]

        sources = [
            prompt_input.format_map(example) if example.get(
                "input", "") != "" else prompt_no_input.format_map(example)
            for example in data
        ]

        label_mapping = {
            "[No Retrieval]": 0,
            "[Retrieval]": 1,
            "[Continue to Use Evidence]": 2,
            "[Irrelevant]": 3,
            "[Relevant]": 4,
            "[Utility:1]": 5,
            "[Utility:2]": 6,
            "[Utility:3]": 7,
            "[Utility:4]": 8,
            "[Utility:5]": 9,
            "[Fully supported]": 10,
            "[Partially supported]": 11,
            "[No support / Contradictory]": 12
        }

        self.labels = [
            label_mapping[example['output']] for example in data]

        logging.warning("Tokenizing inputs... This may take some time...")

        temp = tokenizer(
            sources, padding=True, truncation=True, max_length=512)

        self.input_ids = temp.input_ids

        self.attention_mask = temp.attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], attention_mask=self.attention_mask[i], label=self.labels[i])


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, skip_tokens=None, context_markups=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path,
                                      skip_tokens=skip_tokens, context_markups=context_markups, separated=data_args.separated, is_train=True)

    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path,
                                     skip_tokens=skip_tokens, context_markups=context_markups, separated=data_args.separated, is_train=False)

    return train_dataset, eval_dataset


def tokenize(tokenizer, batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)


def train():

    model_id = "roberta-base"

    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

    dataset = SupervisedDataset(tokenizer)

    test_size = int(len(dataset) * 0.1)

    train_set, val_set = torch.utils.data.random_split(
        dataset, [len(dataset) - test_size, test_size])

    id2label = {i: label for i, label in enumerate(["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]", "[Irrelevant]", "[Relevant]",
                                                    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]", "[Fully supported]", "[Partially supported]", "[No support / Contradictory]"])}
    config = AutoConfig.from_pretrained(model_id)
    config.update({"id2label": id2label})

    # model = RobertaForSequenceClassification.from_pretrained(
    #     model_id, config=config)

    # training_args = TrainingArguments(
    #     output_dir='roberta_model',
    #     num_train_epochs=5,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     evaluation_strategy="epoch",
    #     logging_strategy="steps",
    #     logging_steps=10,
    #     learning_rate=2e-5,
    #     weight_decay=0.01,
    #     warmup_steps=500,
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,
    #     save_total_limit=2,
    # )

    # # Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_set,
    #     eval_dataset=val_set,
    # )

    # trainer.train()

    #################################
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta_model/checkpoint-13200', config=config)

    model.eval()

    data = jload('../gpt4_reward_all_0813_train.json')

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT[
        "prompt_no_input"]

    sources = [
        prompt_input.format_map(example) if example.get(
            "input", "") != "" else prompt_no_input.format_map(example)
        for example in data
    ]

    target = [
        example['output'] for example in data]

    for input, target in zip(sources, target):
        inputs = tokenizer(input, return_tensors="pt",
                           padding=True, truncation=True, max_length=512)

        outputs = model(**inputs)

        predicted_class_index = torch.argmax(outputs.logits).item()

        if id2label[predicted_class_index] != target:
            print("Predicted labels:",
                  id2label[predicted_class_index], "Original label:", target)


if __name__ == "__main__":
    train()

# 3 epochs - 5e-5 8 batch size
# \\{'eval_loss': 0.6455060839653015, 'eval_runtime': 24.3553, 'eval_samples_per_second': 192.648, 'eval_steps_per_second': 12.071, 'epoch': 3.0}
# {'train_runtime': 2016.0028, 'train_samples_per_second': 62.844, 'train_steps_per_second': 3.929, 'train_loss': 0.6093066949434954, 'epoch': 3.0}


# 3 epocgs - 2e-5
#     {'eval_loss': 0.5875992178916931, 'eval_runtime': 24.3875, 'eval_samples_per_second': 192.394, 'eval_steps_per_second': 12.055, 'epoch': 3.0}
# {'train_runtime': 2015.7564, 'train_samples_per_second': 62.851, 'train_steps_per_second': 3.929, 'train_loss': 0.6096030996604399, 'epoch': 3.0}


# 5 epochs 2e-5
#     {'eval_loss': 0.7835614085197449, 'eval_runtime': 24.3554, 'eval_samples_per_second': 192.648, 'eval_steps_per_second': 12.071, 'epoch': 5.0}
# {'train_runtime': 3602.4833, 'train_samples_per_second': 58.614, 'train_steps_per_second': 3.664, 'train_loss': 0.49801656243476, 'epoch': 5.0}
