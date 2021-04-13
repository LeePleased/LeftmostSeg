import random
import os
import time
import numpy as np

import torch


def recursionize(func, query):
    if isinstance(query, (list, tuple, set)):
        return [recursionize(func, i) for i in query]

    return func(query)


def fix_random_seed(state_val):
    random.seed(state_val)
    np.random.seed(state_val)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(state_val)
        torch.cuda.manual_seed_all(state_val)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(state_val)
    torch.random.manual_seed(state_val)


def iob_tagging(segments):
    iob_tags = ["O"] * (segments[-1][1] + 1)

    for start, end, label in segments:
        for i in range(start, end + 1):
            if i == start:
                iob_tags[i] = "B-" + label
            else:
                iob_tags[i] = "I-" + label

    return iob_tags


def f1_scoring(sentences, predictions, golds, script_path):
    fn_out = 'eval_%04d.txt' % random.randint(0, 10000)
    if os.path.isfile(fn_out):
        os.remove(fn_out)

    text_file = open(fn_out, mode='w')
    for i, words in enumerate(sentences):
        tags_1 = golds[i]
        tags_2 = predictions[i]
        for j, word in enumerate(words):
            tag_1 = tags_1[j]
            tag_2 = tags_2[j]
            text_file.write('%s %s %s\n' % (word, tag_1, tag_2))
        text_file.write('\n')
    text_file.close()

    cmd = 'perl %s < %s' % (script_path, fn_out)
    msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
    msg += ''.join(os.popen(cmd).readlines())
    time.sleep(1.0)

    if fn_out.startswith('eval_') and os.path.exists(fn_out):
        os.remove(fn_out)
    return float(msg.split('\n')[3].split(':')[-1].strip())


def input_to_tensor(lexical_vocab, sentences):
    seq_lens = [len(u) for u in sentences]
    max_len = max(seq_lens)

    pad_seq = [u + [lexical_vocab.PAD_SYM] * (max_len - len(u)) for u in sentences]
    var_sent = torch.LongTensor(recursionize(lexical_vocab.index, pad_seq))

    if torch.cuda.is_available():
        var_sent = var_sent.cuda()
    return var_sent, seq_lens


def output_to_tensor(label_vocab, segments):
    spans = [torch.LongTensor([(i, j) for i, j, _ in u]) for u in segments]
    labels = [torch.LongTensor([label_vocab.index(l) for _, _, l in u]) for u in segments]
    remains = [torch.LongTensor([(i, u[-1][1]) for i, j, _ in u]) for u in segments]

    if torch.cuda.is_available():
        spans = [h.cuda() for h in spans]
        labels = [h.cuda() for h in labels]
        remains = [h.cuda() for h in remains]
    return spans, labels, remains


def masking_mat(segments, seq_lens):
    starts = [[i for i, _, _ in u] for u in segments]
    matrices = [torch.LongTensor([[1] * i + [0] * (l - i) for i in u]) for u, l in zip(starts, seq_lens)]

    if torch.cuda.is_available():
        matrices = [u.cuda() for u in matrices]
    return matrices
