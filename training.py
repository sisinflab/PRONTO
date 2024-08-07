import copy

import numpy as np
import torch
import tqdm
from evaluation import compute_metrics
from loss import focal_loss

def beyond_acc(y_true, y_pred):
    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    epsilon = 1e-7
    train_prec = tp / (tp + fp + epsilon)
    train_recall = tp / (tp + fn + epsilon)
    train_f1 = 2 * (train_prec * train_recall) / (train_prec + train_recall + epsilon)

    return train_prec, train_recall, train_f1


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience  # number of times to allow for no improvement before stopping the execution
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_val_loss = -np.inf
        self.best_model = None

    # return True when validation loss is not decreased by the `min_delta` for `patience` times
    def early_stop_check(self, val_loss, model):
        if (val_loss + self.min_delta) > self.min_val_loss:
            self.min_val_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
        elif (val_loss + self.min_delta) <= self.min_val_loss:
            self.counter += 1  # increase the counter if validation loss is not decreased by the min_delta
            if self.counter >= self.patience:
                return True
        return False


def train_model(model, optimizer, X_train, y_train, X_val, y_val, train_prefixes=None, val_prefixes=None, batch_size=64, epochs=50, alpha=0.9):
    early_stopping = EarlyStopping(patience=3, min_delta=0.0)

    training_loader = torch.utils.data.DataLoader(
        range(len(X_train)),
        batch_size=batch_size,
        shuffle=True,
    )


    for t in range(epochs):
        curr_loss = 0
        curr_acc = 0
        curr_f1 = 0
        model.train()
        model.model.eval()
        with tqdm.tqdm(total=int(len(X_train) / batch_size)) as pbar:
            for i, indices in enumerate(training_loader):

                optimizer.zero_grad()

                X_batch = X_train[indices]
                y_batch = y_train[indices]

                if train_prefixes is not None:
                    batch_prefixes = train_prefixes[indices]
                else:
                    batch_prefixes = None

                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(X_batch, prefixes=batch_prefixes, training=True)



                loss = focal_loss(y_pred, y_batch, alpha=alpha, reduction="mean")  # higher weight for positives

                loss += model.reg()

                loss.backward()

                optimizer.step()

                y_labels = (y_pred > 0.5).to(torch.float32)

                train_acc = torch.sum(y_labels == y_batch) / batch_size * 100

                _, _, train_f1 = beyond_acc(y_batch, y_labels)

                curr_loss += loss.item()
                curr_acc += train_acc
                curr_f1 += train_f1
                print(str(curr_loss / (i + 1)))
                pbar.set_postfix(
                    training_loss=str(curr_loss / (i + 1)),
                    train_f1=str(curr_f1 / (i + 1)),
                    curr_acc=str(curr_acc / (i + 1)),
                )

                pbar.update(1)

        model.eval()

        if val_prefixes is None:
            val_prefixes = None

        with torch.no_grad():
            y_val_pred = model(X_val, prefixes=val_prefixes, training=False)

        val_results = compute_metrics(y_val.detach().cpu().numpy(), y_val_pred.detach().cpu().numpy())

        if early_stopping.early_stop_check(val_results["F1 Score"], model):
            print(f"Early stopping performed on epoch {t}")
            break

    model.load_state_dict(early_stopping.best_model)
    return model
