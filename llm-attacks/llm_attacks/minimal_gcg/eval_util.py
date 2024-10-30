from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from tqdm import tqdm

class EvalDataset(Dataset):
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

def eval_perp_based(model, datt, learned_prompt_string, input_string, label_pair):
    """
    Evaluate a prompt-tuned model by comparing the perplexity of two prompt completions.

    Args:
        model: The model to evaluate.
        learned_prompt_string (str): The learned prompt string.
        input_string (str): The input string.
        label_pair (Tuple[str, str]): A pair of labels to compare.
    """

    prompt0 = datt._process_prompt(input_string, learned_prompt_string, label_pair[0])[1]
    prompt1 = datt._process_prompt(input_string, learned_prompt_string, label_pair[1])[1]


    prompt0 = torch.tensor(prompt0).unsqueeze(0).to(model.device)
    prompt1 = torch.tensor(prompt1).unsqueeze(0).to(model.device)

    # print(prompt0)

    # Compute the perplexity of the prompt completions
    with torch.no_grad():
        outputs1 = model(prompt0)
        outputs2 = model(prompt1)


    # calculate cross entropy
    cross_entropy_criterion = torch.nn.CrossEntropyLoss()
    loss1 = cross_entropy_criterion(outputs1.logits.view(-1, model.config.vocab_size), prompt0.view(-1))
    loss2 = cross_entropy_criterion(outputs2.logits.view(-1, model.config.vocab_size), prompt1.view(-1))
    # print(loss1, loss2)

    # print(perplexity1, perplexity2)
    # Compare the perplexities
    if loss1 < loss2:
        return 0
    else:
        return 1


from sklearn.metrics import classification_report
import random
def eval_dataset(model, datt, learned_prompt_string, dataset, label_type="int", device="cuda"):
    """
    Evaluate a prompt-tuned model on a dataset.

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer instance.
        dataset: The dataset to evaluate.
        label_type (str): The type of the labels in the dataset ("int" or "str").
        device: Torch device (e.g., "cuda" or "cpu").
    """
    # Load the model
    # model.to(device)

    pred_list = []
    label_list = []

    # Initialize counters for accuracy calculation
    correct_predictions = 0
    total_samples = len(dataset)
    indices = [random.randint(0, len(dataset)) for i in range(100)]

    # Progress bar for evaluation
    for i in tqdm(indices, desc="Evaluating"):
        # Prepare the input text
        input_text = dataset[i]["sentence"]
        label = dataset[i]["label"]

        if label_type == "int":
            label_pair = [0, 1]
        else:
            label_pair = ["negative", "positive"]

        pred = eval_perp_based(model, datt, learned_prompt_string, input_text, label_pair)
        pred_list.append(pred)
        label_list.append(label)
    
    return classification_report(label_list, pred_list)
