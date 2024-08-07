from datasets import load_dataset
import os
from fet.data import process_sentence
import torch
from config import device
from ProntoLIN import ProntoLIN
from ProntoWS import ProntoWS
from ProntoMAV import ProntoMAV
from ProntoVF import ProntoVF
import tqdm
from pathlib import Path
import argparse


TEMPLATE = "In this sentence, [1] is [MASK] of [2]"
SPECIAL_TOKENS = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script Configuration")

    parser.add_argument("--verbalizer_type", type=str, default="mav", choices=["mav", "vf", "lin", "ws"],
                        help="The name of the output model.")
    parser.add_argument("--model_name", type=str, default="test_model",
                        help="The name of the saved verbalizer model (trained with run_model_train.py).")
    parser.add_argument("--results_name", type=str, default="results",
                        help="The name of the experiment.")
    parser.add_argument("--pretrained_model_name", type=str, default="FacebookAI/roberta-large",
                        help="The name of the pretrained model to use.")
    parser.add_argument("--template_id", type=int, default=0,
                        help="The zero-indexed id of the prompt template to use.")
    parser.add_argument("--experiment_type", type=str, default="coarse", choices=["coarse", "per", "loc", "org"],
                        help="The name of the output model.")

    return parser.parse_args()

args = parse_arguments()

VERBALIZER_TYPE = args.verbalizer_type
RESULTS_NAME = args.results_name
MODEL_NAME = args.model_name
PRETRAINED_MODEL_NAME = args.pretrained_model_name # Huggingface pretrained model
TEMPLATE_ID = args.template_id
EXPERIMENT_TYPE = args.experiment_type

if VERBALIZER_TYPE == 'mav':
    VERBALIZER_CLS = ProntoMAV
elif VERBALIZER_TYPE == 'vf':
    VERBALIZER_CLS = ProntoVF
elif VERBALIZER_TYPE == 'ws':
    VERBALIZER_CLS = ProntoWS
elif VERBALIZER_TYPE == 'lin':
    VERBALIZER_CLS = ProntoLIN

model_parameters = {
    "pretrained_model": PRETRAINED_MODEL_NAME,
    "template": TEMPLATE,
    "special_tokens": SPECIAL_TOKENS,
    "init_token": None,
    "noise_scaling": 0,
    "l1_reg": 0,
}

MODEL_PATH = Path(f"models/{MODEL_NAME}/{MODEL_NAME}-t{TEMPLATE_ID}")

labels = []
labels_file = "coarse_labels.txt"
if EXPERIMENT_TYPE == "coarse":
    labels_file = "coarse_labels.txt"
elif EXPERIMENT_TYPE == "loc":
    labels_file = "labels_location.txt"
elif EXPERIMENT_TYPE == "per":
    labels_file = "labels_person.txt"
elif EXPERIMENT_TYPE == "org":
    labels_file = "labels_organization.txt"

with open(os.path.join("data", "few-nerd", labels_file)) as file:
    for line in file:
        labels.append(line.rstrip())
labels = dict(zip(list(range(1, len(labels) + 1)), labels))

dataset = load_dataset("DFKI-SLT/few-nerd", "supervised", split='test')
samples = []

if EXPERIMENT_TYPE == "coarse":
    dataset_key = "ner_tags"
else:
    dataset_key = "fine_ner_tags"

for i in range(len(dataset)):
    samples.extend(
        process_sentence(
            dataset[i]['tokens'],
            dataset[i][dataset_key],
            neg_tag=0,
            labels=labels
        )
    )


labels_no_other = [label for label in labels.values() if label != "other"]

model = VERBALIZER_CLS.load(
    MODEL_PATH, **model_parameters).to(device)

model.eval()

curr_acc = 0

gt_labels = []
pred_labels = []
with tqdm.tqdm(total=len(samples)) as pbar:
    for i, sample in enumerate(samples):
        try:
            with torch.no_grad():
                pred = model(list(zip([sample[1][0]] * len(labels_no_other), labels_no_other)), prefixes=[sample[0] + " "] * len(labels_no_other))
            if sample[1][1] == labels_no_other[torch.argmax(pred, dim=0).item()]:
                curr_acc += 1
            print(sample[0])
            print(sample[1])
            gt_labels.append(sample[1][1])
            pred_labels.append(labels_no_other[torch.argmax(pred, dim=0).item()])
            print("gt: ", sample[1][1])
            print("pred: ", labels_no_other[torch.argmax(pred, dim=0).item()])
            pbar.set_postfix(
                acc=curr_acc / (i + 1) * 100
            )


            pbar.update(1)
        except:
            continue


with open(f"{RESULTS_NAME}-{MODEL_NAME}.txt", "w") as output:
    output.write(str(list(zip(gt_labels, pred_labels))))


