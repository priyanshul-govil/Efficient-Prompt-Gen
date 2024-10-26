from torch.utils.data import Dataset
from datasets import load_dataset
import torch


class CustomDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        conv_template,
        split,
        dataset_name="stanfordnlp/sst2",
        label_type="int",
        max_length=512,
        debug=False,
    ):
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conv_template = conv_template
        self.adv_string = None
        self.debug = debug
        self.label_type = label_type

    def set_adv_string(self, adv_string):
        self.adv_string = adv_string

    def __len__(self):
        return len(self.dataset)

    def _process_prompt(self, instruction, adv_string, target):
        self.conv_template.append_message(
            self.conv_template.roles[0], f"{instruction} {adv_string}"
        )
        self.conv_template.append_message(self.conv_template.roles[1], f"{target}")
        prompt = self.conv_template.get_prompt()
        if self.conv_template.name == "llama-2":
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            _user_role_slice = slice(0, len(toks))

            self.conv_template.update_last_message(f"{instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            _goal_slice = slice(
                _user_role_slice.stop, max(_user_role_slice.stop, len(toks))
            )

            separator = " " if instruction else ""
            self.conv_template.update_last_message(
                f"{instruction}{separator}{adv_string}"
            )
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            _control_slice = slice(_goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            _assistant_role_slice = slice(_control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            _target_slice = slice(_assistant_role_slice.stop, len(toks) - 2)
            _loss_slice = slice(_assistant_role_slice.stop - 1, len(toks) - 3)

        # convert a python slice into a list of idices
        _user_role_slice = torch.tensor(
            list(range(_user_role_slice.start, _user_role_slice.stop))
        )
        _goal_slice = torch.tensor(list(range(_goal_slice.start, _goal_slice.stop)))
        _control_slice = torch.tensor(
            list(range(_control_slice.start, _control_slice.stop))
        )
        _assistant_role_slice = torch.tensor(
            list(range(_assistant_role_slice.start, _assistant_role_slice.stop))
        )
        _target_slice = torch.tensor(
            list(range(_target_slice.start, _target_slice.stop))
        )
        _loss_slice = torch.tensor(list(range(_loss_slice.start, _loss_slice.stop)))

        return (
            prompt,
            toks,
            _user_role_slice,
            _goal_slice,
            _control_slice,
            _assistant_role_slice,
            _target_slice,
            _loss_slice,
        )

    def __getitem__(self, idx):
        instruction = self.dataset["sentence"][idx]  # assuming sst
        adv_string = self.adv_string
        target = self.dataset["label"][idx]  # assuming sst

        if self.label_type != "int":
            if target == 0:
                target = "negative"
            else:
                target = "positive"

        print(target)
        # print("GETITEM")
        (
            prompt,
            toks,
            _user_role_slice,
            _goal_slice,
            _control_slice,
            _assistant_role_slice,
            _target_slice,
            _loss_slice,
        ) = self._process_prompt(instruction, adv_string, target)

        attention_mask = torch.ones(len(toks), dtype=torch.long)

        ret_dict = {
            "input_ids": toks,
            # "_user_role_slice": _user_role_slice,
            # "_goal_slice": _goal_slice,
            "_control_slice": _control_slice,
            # "_assistant_role_slice": _assistant_role_slice,
            "_target_slice": _target_slice,
            "_loss_slice": _loss_slice,
            "attention_mask": attention_mask,
            "pad_token": self.tokenizer.pad_token_id,
        }

        if self.debug:
            ret_dict["prompt"] = prompt
            ret_dict["instruction"] = instruction
            ret_dict["adv_string"] = adv_string
            ret_dict["target"] = target

        return ret_dict


def collate_fn(batch):
    # Pad each input and label sequence to the length of the longest in the batch
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    padding_token = batch[0]["pad_token"]
    # Pad the sequences in each list to the longest in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=padding_token
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    ret = {}
    # copy everything else appearing in each item of the batch
    for key in batch[0].keys():
        if key not in ["input_ids", "attention_mask", "pad_token"]:
            print(key)
            ret[key] = torch.stack([item[key] for item in batch])

    ret.update(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )

    return ret
