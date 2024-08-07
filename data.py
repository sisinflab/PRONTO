import json
import os.path
from transformers import AutoTokenizer
import numpy as np
import random
import torch
from config import device
import tqdm
import pandas as pd


def check_for_length_limit(desc_dict, pretrained_model="FacebookAI/roberta-large", max_tokens=400):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    out_dict = {}
    for k,v in desc_dict.items():
        length = len(tokenizer(v, add_special_tokens=False).input_ids)
        if length <= max_tokens:
            out_dict.setdefault(k, v)
        else:
            continue

    return out_dict


def preprocess_desc_dict(desc_dict, concat_symbol=": "):
    out_dict = {}
    for k, v in desc_dict.items():
        out_dict.setdefault(k, f"{k}{concat_symbol}{v}")
    return out_dict


def get_desc(max_tokens=50):
    classes_desc = pd.read_csv("./data/classes.tsv", sep="\t", header=0)
    instances_desc = pd.read_csv("./data/instances.tsv", sep="\t", header=0)
    classes_desc = classes_desc[["Label", "Description"]].set_index("Label").to_dict()["Description"]
    instances_desc = instances_desc[["Label", "Description"]].set_index("Label").to_dict()["Description"]
    print(len(instances_desc))
    classes_desc = preprocess_desc_dict(classes_desc, concat_symbol=": ")
    classes_desc = check_for_length_limit(classes_desc, max_tokens=max_tokens)
    instances_desc = check_for_length_limit(instances_desc, max_tokens=max_tokens)
    print(len(instances_desc))


    return classes_desc | instances_desc


def get_data(train_test_split=0.5, test_val_split=0.99, capitalize=False, reversed_test=False, with_desc=False, unseen_instances=0):
    with open('./data/class.json') as f:
        data = json.load(f)

    data_pos = []
    data_neg = []
    hard_negatives = []
    classes = []
    instances = []
    for entry in data:
        pairs = []
        classes.append(entry['rdfs:label'])
        if 'rdfs:subClassOf' in entry:
            for parent in entry['rdfs:subClassOf']:
                pairs.append((entry['rdfs:label'], parent))
            for parent in [*entry['rdfs:subClassOf'], entry['rdfs:label']]:
                for child in entry['is rdf:type of']:
                    pairs.append((child, parent))
                    instances.append(child)
        data_pos.extend(pairs)

    data_pos = np.unique(np.array(data_pos), axis=0)
    classes = np.unique(np.array(classes), axis=0)
    instances = np.unique(np.array(instances), axis=0)

    classes_desc = {}
    instances_desc = {}
    #np.savetxt("instances.txt", instances, fmt="%s")
    #np.savetxt("classes.txt", classes, fmt="%s")

    if with_desc:
        descriptions_map = get_desc()

    if os.path.exists("./data/data_neg.txt") and os.path.exists("./data/hard_negatives.txt"):
        data_neg = np.loadtxt("./data/data_neg.txt", delimiter="\t", dtype=str)
        hard_negatives = np.loadtxt("./data/hard_negatives.txt", delimiter="\t", dtype=str)
    else:
        t_ent = np.unique(np.array(data_pos[:, 1]))
        for pair in tqdm.tqdm(data_pos, desc="Generating negative samples.."):
            for _ in range(2):
                gold_classes = np.unique(data_pos[np.where(data_pos[:, 0] == pair[0])[0]][:, 1])
                gold_root = np.unique(data_pos[np.where(np.isin(data_pos[:, 0],gold_classes))[0]][:, 1])
                candidate_classes = np.intersect1d(np.unique(data_pos[np.where(np.isin(data_pos[:, 1], gold_root))[0]][:, 0]), t_ent)
                #neg_pair = [pair[0], np.random.choice(list(np.setdiff1d(candidate_classes, np.unique(data_pos[np.where(data_pos[:, 0] == pair[0])[0]][:, 1]))))]
                candidate_classes = tuple(np.setdiff1d(candidate_classes, np.unique(data_pos[np.where(data_pos[:, 0] == pair[0])[0]][:, 1])))
                # hard negatives
                if len(candidate_classes) == 0:
                    break
                hard_neg_pair = (pair[0], np.random.choice(candidate_classes))
                data_neg.append(hard_neg_pair)
                hard_negatives.append(hard_neg_pair)
            for _ in range(1):
                candidate_soft_classes = data_pos[np.where(data_pos[:, 0] != pair[0])][:, 1]
                candidate_soft_classes = candidate_soft_classes[np.isin(candidate_soft_classes, gold_classes, invert=True)]
                data_neg.append(
                    (pair[0], np.random.choice(candidate_soft_classes))
                )
            data_neg.append((pair[1], pair[0]))  # soft negative

        data_neg = np.unique(np.array(data_neg), axis=0)
        hard_negatives = np.unique(np.array(hard_negatives), axis=0)

        np.savetxt("./data/data_neg.txt", data_neg, delimiter="\t", fmt="%s")
        np.savetxt("./data/hard_negatives.txt", hard_negatives, delimiter="\t", fmt="%s")

    X = np.concatenate([data_pos, data_neg], axis=0)

    if capitalize:
        X[:, 1] = np.char.capitalize(X[:, 1])

    y = np.concatenate([np.ones(len(data_pos)), np.zeros(len(data_neg))], axis=0)

    total_pos_samples = len(data_pos)
    split_index = int(train_test_split * total_pos_samples)

    pos_indices = np.arange(total_pos_samples)
    np.random.shuffle(pos_indices)

    total_samples = len(X)

    total_neg_samples = total_samples - total_pos_samples
    neg_indices = np.arange(total_neg_samples)
    np.random.shuffle(neg_indices)
    neg_indices += total_pos_samples

    # Split the indices into training and testing sets
    train_pos_indices = pos_indices[:split_index]
    non_train_pos_indices = pos_indices[split_index:]

    neg_split_index = total_pos_samples - split_index  # number of positives in non_train
    non_train_neg_indices = neg_indices[:neg_split_index]
    train_neg_indices = neg_indices[neg_split_index:]

    train_indices = np.concatenate([train_pos_indices, train_neg_indices])
    non_train_indices = np.concatenate([non_train_pos_indices, non_train_neg_indices])

    np.random.shuffle(train_indices)
    np.random.shuffle(non_train_indices)

    total_test_samples = len(non_train_indices)
    test_split_index = int(test_val_split * total_test_samples)

    test_indices = non_train_indices[:test_split_index]
    val_indices = non_train_indices[test_split_index:]

    # Use the indices to split X and y
    X_train, X_test, X_val = X[train_indices], X[test_indices], X[val_indices]
    y_train, y_test, y_val = y[train_indices], y[test_indices], y[val_indices]

    test_reversed_neg = np.where((data_pos[:, [1, 0]] == X_test[:, None]).all(-1).any(-1))

    if reversed_test:

        X_test_reversed_negatives = X_test[test_reversed_neg]
        #y_test_reversed_negatives = y_test[test_reversed_neg]
        X_test = np.delete(X_test, np.where(y_test == 0), axis=0)
        y_test = np.concatenate([[1] * len(X_test), [0] * len(X_test_reversed_negatives)])
        X_test = np.concatenate([X_test, X_test_reversed_negatives], axis=0)
       # test_negatives = X_test[np.where(y_test == 0)]
    else:
        test_negatives = X_test[np.where(y_test == 0)]
        X_test = np.delete(X_test, np.where(y_test == 0), axis=0)
        test_hard_negatives = []
        for hn in test_negatives:
            if (hard_negatives == hn).all(-1).sum() > 0:
                test_hard_negatives.append(hn)

        test_hard_negatives = np.array(test_hard_negatives)
        y_test = np.concatenate([[1]*len(X_test), [0]*len(test_hard_negatives)])
        X_test = np.concatenate([X_test, test_hard_negatives], axis=0)

        np.savetxt("./test_pairs.txt", X_test, delimiter="\t", fmt="%s")
        np.savetxt("./test_pairs_gt.txt", y_test, delimiter="\t", fmt="%s")

    # unseen entities
    if unseen_instances > 0:

        test_entities = np.intersect1d(instances, np.unique(X_test[:, 0]))
        np.random.shuffle(test_entities)
        test_entities = test_entities[:int(unseen_instances*len(test_entities))]
        y_train = y_train[np.isin(X_train, test_entities, invert=True).all(1)]
        X_train = X_train[np.isin(X_train, test_entities, invert=True).all(1)]


    y_train = torch.tensor(y_train).to(torch.float32).to(device)
    y_test = torch.tensor(y_test).to(torch.float32).to(device)
    y_val = torch.tensor(y_val).to(torch.float32).to(device)


    if with_desc:
        train_prefixes = []
        test_prefixes = []
        val_prefixes = []
        for x in X_train:
            h_desc = descriptions_map.get(x[0], "") + " "
            t_desc = descriptions_map.get(x[1], "") + " "
            train_prefixes.append(f"{h_desc}{t_desc}")
        for x in X_test:
            h_desc = descriptions_map.get(x[0], "") + " "
            t_desc = descriptions_map.get(x[1], "") + " "
            test_prefixes.append(f"{h_desc}{t_desc}")
        for x in X_val:
            h_desc = descriptions_map.get(x[0], "") + " "
            t_desc = descriptions_map.get(x[1], "") + " "
            val_prefixes.append(f"{h_desc}{t_desc}")
        train_prefixes = np.array(train_prefixes)
        test_prefixes = np.array(test_prefixes)
        val_prefixes = np.array(val_prefixes)
    else:
        train_prefixes = None
        test_prefixes = None
        val_prefixes = None

    return (X_train, y_train), (X_test, y_test), (X_val, y_val), (train_prefixes, test_prefixes, val_prefixes)


if __name__ == "__main__":

    descriptions = get_desc()
    print(descriptions)
