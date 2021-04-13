import argparse
import os
import json

import torch
from torch.optim import Adam

from misc import fix_random_seed
from utils import LexicalAlphabet, LabelAlphabet
from model import LeftmostSeg
from utils import corpus_to_iterator
from utils import Procedure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", "-dd", type=str, required=True)
    parser.add_argument("--check_dir", "-cd", type=str, required=True)
    parser.add_argument("--script_path", "-sp", type=str, required=True)
    parser.add_argument("--random_state", "-rs", type=int, default=0)
    parser.add_argument("--epoch_num", "-en", type=int, default=100)
    parser.add_argument("--batch_size", "-bs", type=int, default=8)

    parser.add_argument("--word_embedding_dim", "-wed", type=int, default=128)
    parser.add_argument("--label_embedding_dim", "-led", type=int, default=32)
    parser.add_argument("--enc_hidden_dim", "-ehd", type=int, default=128)
    parser.add_argument("--dec_hidden_dim", "-dhd", type=int, default=512)
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.3)

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=True, ensure_ascii=False), end="\n\n")

    fix_random_seed(args.random_state)

    lexical_vocab = LexicalAlphabet()
    label_vocab = LabelAlphabet()

    train_loader = corpus_to_iterator(os.path.join(args.data_dir, "train.txt"),
                                      args.batch_size, True,
                                      lexical_vocab, label_vocab)
    dev_loader = corpus_to_iterator(os.path.join(args.data_dir, "dev.txt"), args.batch_size, False)
    test_loader = corpus_to_iterator(os.path.join(args.data_dir, "test.txt"), args.batch_size, False)

    model = LeftmostSeg(lexical_vocab, label_vocab, args.word_embedding_dim,
                        args.label_embedding_dim, args.enc_hidden_dim,
                        args.dec_hidden_dim, args.dropout_rate)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = Adam(model.parameters(), weight_decay=1e-6)

    best_dev = 0.0
    save_path = os.path.join(args.check_dir, "model.pt")
    if not os.path.exists(args.check_dir):
        os.makedirs(args.check_dir)

    for epoch_idx in range(0, args.epoch_num + 1):
        train_loss, train_time = Procedure.train(model, train_loader, optimizer)
        print("[Epoch {:3d}] loss on train set is {:.5f} using {:.3f} secs".format(epoch_idx, train_loss, train_time))

        dev_f1, dev_time = Procedure.evaluate(model, dev_loader, args.script_path)
        print("(Epoch {:3d}) f1 score on dev set is {:.5f} using {:.3f} secs".format(epoch_idx, dev_f1, dev_time))

        test_f1, test_time = Procedure.evaluate(model, test_loader, args.script_path)
        print("{{Epoch {:3d}}} f1 score on test set is {:.5f} using {:.3f} secs".format(epoch_idx, test_f1, test_time))

        if dev_f1 > best_dev:
            best_dev = dev_f1

            print("\n<Epoch {:3d}> save the model with test score, {:.5f}, in terms of dev".format(epoch_idx, test_f1))
            torch.save(model, save_path)
        print(end="\n\n")
