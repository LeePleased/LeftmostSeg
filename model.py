import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from misc import input_to_tensor
from misc import output_to_tensor
from misc import masking_mat


class LeftmostSeg(nn.Module):

    _INFINITY_VAL = -1e8
    
    def __init__(self,
                 lexical_vocab,
                 label_vocab,
                 word_embedding_dim,
                 label_embedding_dim,
                 enc_hidden_dim,
                 dec_hidden_dim,
                 dropout_rate):
        super(LeftmostSeg, self).__init__()

        self._lexical_vocab = lexical_vocab
        self._label_vocab = label_vocab

        self._lexical_embedding = nn.Embedding(len(lexical_vocab), word_embedding_dim, padding_idx=0)
        self._lexical_embedding.weight.data[1, :] = 0.0
        self._label_embedding = nn.Embedding(len(label_vocab), label_embedding_dim)
        self._sos_embedding = nn.Parameter(torch.randn(1, enc_hidden_dim * 3 + label_embedding_dim))
        self._sos_embedding.requires_grad = True

        self._encoder = BiLSTMEnc(word_embedding_dim, enc_hidden_dim, dropout_rate)
        self._decoder = UniLSTMDec(enc_hidden_dim * 6 + label_embedding_dim, dec_hidden_dim, dropout_rate)

        self._span_projector = MLP(dec_hidden_dim, enc_hidden_dim * 3, dropout_rate)
        self._label_projector = MLP(dec_hidden_dim + enc_hidden_dim * 3, label_embedding_dim, dropout_rate)
        self._nll_loss = nn.NLLLoss()

    def forward(self, serial_seq, seq_lens):
        unit_h = self._lexical_embedding(serial_seq)
        rnn_h = self._encoder(unit_h, seq_lens)

        batch_size, unit_num, hidden_dim = rnn_h.size()
        row_h = rnn_h.unsqueeze(2).expand(batch_size, unit_num, unit_num, hidden_dim)
        column_h = rnn_h.unsqueeze(1).expand_as(row_h)
        return torch.cat([column_h, column_h - row_h, row_h], dim=-1)

    def estimate(self, sentences, segments):
        idx_seq, seq_lens = input_to_tensor(self._lexical_vocab, sentences)
        spans, labels, remains = output_to_tensor(self._label_vocab, segments)
        masks = masking_mat(segments, seq_lens)

        phrase_reprs = self(idx_seq, seq_lens)
        batch_size, _, _, _ = phrase_reprs.size()
        avg_ce_loss = 0.0

        for case_i in range(0, batch_size):
            p_table = phrase_reprs[case_i, :seq_lens[case_i], :seq_lens[case_i]]

            sp_hidden = p_table[spans[case_i][:, 0], spans[case_i][:, 1]]
            lb_hidden = self._label_embedding(labels[case_i])
            seg_hidden = torch.cat([sp_hidden, lb_hidden], dim=-1)
            prior_seg = torch.cat([self._sos_embedding, seg_hidden[:-1, :]], dim=0)

            rm_hidden = p_table[remains[case_i][:, 0], remains[case_i][:, 1]]
            info_h = torch.cat([prior_seg, rm_hidden], dim=-1)
            rnn_hidden, _, _ = self._decoder(info_h)

            sp_feeds = self._span_projector(rnn_hidden)
            candidates, ground_truths = p_table[spans[case_i][:, 0]], spans[case_i][:, 1]
            s_score = torch.bmm(sp_feeds.unsqueeze(-2), candidates.transpose(-1, -2)).squeeze(1)
            s_score.masked_fill_(masks[case_i] == 1, self._INFINITY_VAL)
            s_penalty = self._nll_loss(torch.log_softmax(s_score, dim=-1), ground_truths)

            lb_feed = self._label_projector(torch.cat([sp_hidden, rnn_hidden], dim=-1))
            l_score = torch.matmul(lb_feed, self._label_embedding.weight.transpose(0, 1))
            lb_penalty = self._nll_loss(torch.log_softmax(l_score, dim=-1), labels[case_i])

            paired_loss = s_penalty + lb_penalty
            avg_ce_loss += paired_loss / batch_size

        return avg_ce_loss

    def predict(self, sentences):
        idx_seq, seq_lens = input_to_tensor(self._lexical_vocab, sentences)
        phrase_reprs = self.forward(idx_seq, seq_lens)

        batch_size = len(sentences)
        outputs = []
        for case_i in range(0, batch_size):
            segments, start, length = [], 0, seq_lens[case_i]
            p_table = phrase_reprs[case_i, :length, :length]
            prior_seg = self._sos_embedding
            prior_state, prior_cell = None, None

            while start < length:
                remain_h = p_table[start, length - 1]
                feed_h = torch.cat([prior_seg, remain_h.unsqueeze(0)], dim=-1)
                rnn_h, prior_state, prior_cell = self._decoder(feed_h, prior_state, prior_cell)

                sp_input = self._span_projector(rnn_h)
                candidates = p_table[start, :]
                s_score = torch.matmul(sp_input, candidates.transpose(0, 1)).squeeze(0)
                s_score[:start] = self._INFINITY_VAL
                end = s_score.topk(1)[1].item()

                mention = p_table[start, end]
                lb_input = self._label_projector(torch.cat([mention, rnn_h.squeeze(0)], dim=-1))
                lb_score = torch.matmul(self._label_embedding.weight, lb_input)
                idx_l = lb_score.topk(1)[1].item()

                segments.append((start, end, self._label_vocab.get(idx_l)))
                start = end + 1
                prior_seg = torch.cat([mention, self._label_embedding.weight[idx_l]], dim=-1).unsqueeze(0)
            outputs.append(segments)

        return outputs


class BiLSTMEnc(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate):
        super(BiLSTMEnc, self).__init__()

        self._rnn = nn.LSTM(input_dim, output_dim // 2, batch_first=True, bidirectional=True)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, state_h, seq_lens):
        sent_num = len(seq_lens)
        sort_idx_list = sorted(range(0, sent_num), key=seq_lens.__getitem__, reverse=True)

        sort_len_list = [seq_lens[i] for i in sort_idx_list]
        sort_h = self._dropout(state_h)[sort_idx_list, :, :]
        pack_h = pack_padded_sequence(sort_h, sort_len_list, batch_first=True)
        rnn_h = pad_packed_sequence(self._rnn(pack_h)[0], batch_first=True)[0]

        rev_idx_list = [-1 for _ in range(0, sent_num)]
        for j in range(0, sent_num):
            rev_idx_list[sort_idx_list[j]] = j
        rev_len_list = [sort_len_list[i] for i in rev_idx_list]

        assert rev_len_list == seq_lens
        return rnn_h[rev_idx_list, :, :]


class UniLSTMDec(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate):
        super(UniLSTMDec, self).__init__()

        self._rnn = nn.LSTM(input_dim, output_dim, batch_first=True)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden, state=None, cell=None):
        drop_h = self._dropout(hidden).unsqueeze(0)

        if state is None or cell is None:
            rnn_h, (nxt_s, nxt_c) = self._rnn(drop_h)
        else:
            state = state.unsqueeze(0)
            cell = cell.unsqueeze(0)
            rnn_h, (nxt_s, nxt_c) = self._rnn(drop_h, (state, cell))
        return rnn_h.squeeze(0), nxt_s.squeeze(0), nxt_c.squeeze(0)


class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate):
        super(MLP, self).__init__()

        self._computations = nn.Sequential(nn.Dropout(dropout_rate),
                                           nn.Linear(input_dim, output_dim),
                                           nn.LeakyReLU())

    def forward(self, hidden):
        return self._computations(hidden)
