from ProntoLIN import ProntoLIN
from ProntoWS import ProntoWS
from ProntoMAV import ProntoMAV
from ProntoVF import ProntoVF
import torch
from config import device
from evaluation import evaluate
from utils import enable_determinism
from data import get_data
import numpy as np
from training import train_model
from pathlib import Path
import json
import os
import argparse


TEMPLATES = [
    "[1] is [MASK] of [2]",
    "[1] is a [MASK] of [2]",
    "[1] [MASK] [2]",
    "[1] is [MASK] [2]",
    "[1] [MASK] of [2]",
    "[1] is [R1][MASK][R2] of [2]",
    "[1] [R1][MASK][R2] [2]",
]

INIT_TOKENS = [
    "type",
    "type",
    "is",
    "a",
    "instance",
    None,
    None,
]

SPECIAL_TOKENS = [
    None,
    None,
    None,
    None,
    None,
    ["[R1]", "[R2]"],
    ["[R1]", "[R2]"],
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script Configuration")
    parser.add_argument("--verbalizer_type", type=str, default="mav", choices=["mav", "vf", "lin", "ws"],
                        help="The name of the output model.")
    parser.add_argument("--model_name", type=str, default="test_model",
                        help="The name of the output model.")
    parser.add_argument("--pretrained_model_name", type=str, default="FacebookAI/roberta-large",
                        help="The name of the pretrained model to use.")
    parser.add_argument("--no_save_models", action="store_false",
                        help="Whether to save the models.")
    parser.add_argument("--no_save_results", action="store_false",
                        help="Whether to save the results.")
    parser.add_argument("--load_models", action="store_true",
                        help="Whether to load existing models.")
    parser.add_argument("--no_train_models", action="store_false",
                        help="Whether to train the models.")
    parser.add_argument("--with_context", action="store_true",
                        help="Whether to use textual descriptions.")
    parser.add_argument("--reversed_test", action="store_true",
                        help="Whether to perform the positive-reverse negatives test.")
    parser.add_argument("--unseen_instances", type=float, default=0,
                        help="Percentage of instances in the test set to drop associated triples from the train set.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Alpha parameter value for the focal loss.")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay value.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate.")
    parser.add_argument("--l1_reg", type=float, default=0,
                        help="L1 regularization value.")
    parser.add_argument("--noise_scaling", type=float, default=0,
                        help="Noise scaling value (DEPRECATED).")

    return parser.parse_args()


enable_determinism()
args = parse_arguments()

VERBALIZER_TYPE = args.verbalizer_type
MODEL_NAME = args.model_name
PRETRAINED_MODEL_NAME = args.pretrained_model_name
SAVE_MODELS = args.no_save_models
SAVE_RESULTS = args.no_save_results
LOAD_MODELS = args.load_models
TRAIN_MODELS = args.no_train_models
WITH_CONTEXT = args.with_context
REVERSED_TEST = args.reversed_test
UNSEEN_INSTANCES = args.unseen_instances
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
ALPHA = args.alpha
WEIGHT_DECAY = args.weight_decay
LEARNING_RATE = args.learning_rate
L1_REG = args.l1_reg
NOISE_SCALING = args.noise_scaling

print(f"Verbalizer Type: {VERBALIZER_TYPE}")
print(f"Model Name: {MODEL_NAME}")
print(f"Pretrained Model Name: {PRETRAINED_MODEL_NAME}")
print(f"Save Models: {SAVE_MODELS}")
print(f"Save Results: {SAVE_RESULTS}")
print(f"Load Models: {LOAD_MODELS}")
print(f"Train Models: {TRAIN_MODELS}")
print(f"With Context: {WITH_CONTEXT}")
print(f"Reversed Test: {REVERSED_TEST}")
print(f"Unseen Instances: {UNSEEN_INSTANCES}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Alpha: {ALPHA}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"L1 Regularization: {L1_REG}")
print(f"Noise Scaling: {NOISE_SCALING}")

if VERBALIZER_TYPE == 'mav':
    VERBALIZER_CLS = ProntoMAV
elif VERBALIZER_TYPE == 'vf':
    VERBALIZER_CLS = ProntoVF
elif VERBALIZER_TYPE == 'ws':
    VERBALIZER_CLS = ProntoWS
elif VERBALIZER_TYPE == 'lin':
    VERBALIZER_CLS = ProntoLIN

train_data, test_data, val_data, prefixes = get_data(with_desc=WITH_CONTEXT, reversed_test=REVERSED_TEST, unseen_instances=UNSEEN_INSTANCES)

X_train, y_train = train_data
X_test, y_test = test_data
X_val, y_val = val_data
train_prefixes, test_prefixes, val_prefixes = prefixes

print(len(y_val))

assert len(INIT_TOKENS) == len(TEMPLATES)

for template_id, template in enumerate(TEMPLATES):
    model_parameters = {
        "pretrained_model": PRETRAINED_MODEL_NAME,
        "template": template,
        "special_tokens": SPECIAL_TOKENS[template_id],
        "init_token": INIT_TOKENS[template_id],
        "noise_scaling": NOISE_SCALING,
        "l1_reg": L1_REG,
    }
    model_base_path = Path(f"models/{MODEL_NAME}/{MODEL_NAME}-t{template_id}")

    if LOAD_MODELS:
        model = VERBALIZER_CLS.load(
            model_base_path,
            **model_parameters
        ).to(device)
        best_model = model

    else:  # train model

        model = VERBALIZER_CLS(
            **model_parameters
        ).to(device)

        if TRAIN_MODELS:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )

            best_model = train_model(
                model,
                optimizer,
                X_train,
                y_train,
                X_val,
                y_val,
                train_prefixes = train_prefixes,
                val_prefixes = val_prefixes,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                alpha=ALPHA
            )
        else:
            best_model = model

    if SAVE_MODELS:
        model_base_path.mkdir(parents=True, exist_ok=True)
        best_model.save(model_base_path)
        # torch.save(best_model, os.path.join(model_base_path, f"{MODEL_NAME}-t{template_id}.pt"))

    print(best_model.get_top_tokens())
    # perform evaluation on current model-template
    evaluation_res = evaluate(best_model, X_test, y_test, prefixes=test_prefixes)

    print(f"Results for model {MODEL_NAME}")
    print(f"(Template: {template})")
    print(evaluation_res)

    if SAVE_RESULTS:
        model_base_path.mkdir(parents=True, exist_ok=True)
        model.generate_word_cloud(
            os.path.join(model_base_path, f"wc-{MODEL_NAME}-t{template_id}.png")
        )
        with open(
            os.path.join(model_base_path, f"results-{MODEL_NAME}-t{template_id}.json"),
            "w",
        ) as f:
            json.dump(evaluation_res, f, indent=4, default=str)