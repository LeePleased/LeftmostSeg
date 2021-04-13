import codecs
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from misc import iob_tagging
from misc import recursionize
from misc import f1_scoring


class LexicalAlphabet(object):

    PAD_SYM, UNK_SYM = "[PAD]", "[UNK]"

    def __init__(self):
        super(LexicalAlphabet, self).__init__()

        self._idx_to_item = []
        self._item_to_idx = {}

        self.add(LexicalAlphabet.PAD_SYM)
        self.add(LexicalAlphabet.UNK_SYM)

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def index(self, item):
        try:
            return self._item_to_idx[item]
        except KeyError:
            return self._item_to_idx[self.UNK_SYM]

    def __len__(self):
        return len(self._idx_to_item)


class LabelAlphabet(object):

    def __init__(self):
        super(LabelAlphabet, self).__init__()

        self._idx_to_item = []
        self._item_to_idx = {}

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item):
        return self._item_to_idx[item]

    def __len__(self):
        return len(self._idx_to_item)


def corpus_to_iterator(file_path, batch_size, if_shuffle, lexical_vocab=None, label_vocab=None):
    mentions = []

    with codecs.open(file_path, "r", "utf-8") as fr:
        buffer = []

        for line in fr:
            items = line.strip().split()

            if len(items) == 2:
                buffer.append(items)
            elif len(items) == 0:
                if len(buffer) > 0:
                    mentions.append(buffer)
                    buffer = []
            else:
                assert Exception("Data Format Error!")

    sentences, segments = [], []
    for case in mentions:
        sentences.append([])
        segments.append([])

        pointer = 0
        for phrase, tag in case:
            sentences[-1].extend(list(phrase))
            segments[-1].append((pointer, pointer + len(phrase) - 1, tag))
            pointer = pointer + len(phrase)

    if lexical_vocab is not None:
        recursionize(lexical_vocab.add, sentences)

    if label_vocab is not None:
        for case in segments:
            for _, _, lb in case:
                label_vocab.add(lb)

    class _Dataset(Dataset):

        def __init__(self, *args):
            self._args = args

        def __getitem__(self, item):
            return [s[item] for s in self._args]

        def __len__(self):
            return len(self._args[0])

    wrap_data = _Dataset(sentences, segments)
    return DataLoader(wrap_data, batch_size, if_shuffle, collate_fn=lambda x: list(zip(*x)))


class Procedure(object):

    @staticmethod
    def train(model, dataset, optimizer):
        model.train()
        time_start, total_loss = time.time(), 0.0

        for sentences, segments in tqdm(dataset, ncols=50):
            penalty = model.estimate(sentences, segments)
            total_loss += penalty.item()

            optimizer.zero_grad()
            penalty.backward()
            optimizer.step()

        time_pass = time.time() - time_start
        return total_loss, time_pass

    @staticmethod
    def evaluate(model, dataset, script_path):
        model.eval()

        time_start = time.time()
        sentences, predictions, oracles = [], [], []

        for seqs, segments in tqdm(dataset, ncols=50):
            with torch.no_grad():
                predictions.extend([iob_tagging(u) for u in model.predict(seqs)])
            oracles.extend([iob_tagging(u) for u in segments])
            sentences.extend(seqs)

        out_f1 = f1_scoring(sentences, predictions, oracles, script_path)
        return out_f1, time.time() - time_start
