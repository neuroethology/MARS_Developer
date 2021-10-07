from __future__ import division
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
import random
import pdb

def score_info(y, y_pred, vocab):
    precision, recall, fscore, _ = score(y, y_pred)
    beh_names = list(vocab.keys())
    beh_list = [vocab[b] for b in beh_names]
    print(f"{' ' : >12}", end=" ")
    for i in list(np.unique(y)):
        ind = beh_list.index(i)
        print(f"{beh_names[ind] : ^9}", end=" ")
    print(" ")
    print(f"{'Precision:' : >12}", end=" ")
    for i in precision:
        print(f"{np.round(i, 3) : ^9}", end=" ")
    print(" ")
    print(f"{'Recall:' : >12}", end=" ")
    for i in recall:
        print(f"{np.round(i, 3) : ^9}", end=" ")
    print(" ")
    print(f"{'F1 Score:' : >12}", end=" ")
    for i in fscore:
        print(f"{np.round(i, 3) : ^9}", end=" ")
    print(" ")
    return precision, recall, fscore


def prf_metrics(y_tr_beh, pd_class, beh):
    eps = np.spacing(1)
    pred_pos = np.where(pd_class == 1)[0]
    true_pred = np.where(y_tr_beh[pred_pos] == 1)[0]
    true_pos = np.where(y_tr_beh == 1)[0]
    pred_true = np.where(pd_class[true_pos] == 1)[0]

    n_pred_pos = len(pred_pos)
    n_true_pred = len(true_pred)
    n_true_pos = len(true_pos)
    n_pred_true = len(pred_true)

    precision = n_true_pred / (n_pred_pos+eps)
    recall = n_pred_true / (n_true_pos+eps)
    f_measure = 2 * precision * recall / (precision + recall+np.spacing(1))
    print('P: %5.4f, R: %5.4f, F1: %5.4f    %s' % (precision, recall, f_measure, beh))
    return precision, recall, f_measure


def shuffle_fwd(L):
    idx = list(range(L.shape[0]))
    random.shuffle(idx)
    return L[idx], idx


def shuffle_back(L,idx):
    L_out = np.zeros(L.shape)
    for i,j in enumerate(idx):
        L_out[j] = L[i]
    return L_out


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()


class Batch:
    def __init__(self, iterable, condition=(lambda x: True), limit=None):
        self.iterator = iter(iterable)
        self.condition = condition
        self.limit = limit
        try:
            self.current = next(self.iterator)
        except StopIteration:
            self.on_going = False
        else:
            self.on_going = True

    def group(self):
        yield self.current
        # start enumerate at 1 because we already yielded the last saved item
        for num, item in enumerate(self.iterator, 1):
            self.current = item
            if num == self.limit or self.condition(item):
                break
            yield item
        else:
            self.on_going = False

    def __iter__(self):
        while self.on_going:
            yield self.group()

def get_onehot_labels(labels, behavior):
    onehot_labels = np.array([1 if behavior == label else 0 for label in labels])
    return onehot_labels


def flatten(*arg):
    """Flattens everything into a list, by decomposing tuples and lists into their singular parts recursively."""
    return (result for element in arg
            for result in (flatten(*element) if isinstance(element, (tuple, list))
                           else (element,)))
