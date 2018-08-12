import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

from Net import BiLstmSentenceEncoder

# Modules
class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)

class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass

class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass

class ScaledDotProductAttention(nn.Module):

    def __init__(self, model_dim, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_constant = np.power(model_dim, 0.5)
        self.dropout_layer = nn.Dropout(attention_dropout)
        self.softmax_layer = BottleSoftmax()

    def forward(self, q, k, v, mask=None):
        attention_matrix = torch.bmm(q, k.transpose(1, 2)) / self.scale_constant

        if mask is not None:
            attention_matrix.data.masked_fill_(mask, -float('inf'))

        attention_matrix = self.softmax_layer(attention_matrix)
        attention_matrix = self.dropout_layer(attention_matrix)

        output = torch.bmm(attention_matrix, v)

        return output

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

# Sublayer
class MultiHeadAttentionUnit(nn.Module):

    def __init__(self, head_num, model_dim, k_dim, v_dim, dropout):
        super(MultiHeadAttentionUnit, self).__init__()

        self.head_num = head_num
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.w_qs = nn.Parameter(torch.FloatTensor(head_num, model_dim, k_dim))
        self.w_ks = nn.Parameter(torch.FloatTensor(head_num, model_dim, k_dim))
        self.w_vs = nn.Parameter(torch.FloatTensor(head_num, model_dim, v_dim))

        self.attention_layer = ScaledDotProductAttention(model_dim)
        self.norm_layer = LayerNormalization(model_dim)
        self.linear_layer = BottleLinear(head_num*v_dim, model_dim) # nn.Linear(head_num * v_dim, model_dim, bias=True)

        self.dropout_layer = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q, k, v, attention_mask=None):
        k_dim = self.k_dim
        v_dim = self.v_dim
        head_num = self.head_num

        res = q

        batch_size, q_len, model_dim = q.size()
        batch_size, k_len, model_dim = k.size()
        batch_size, v_len, model_dim = v.size()

        multi_q = q.repeat(head_num, 1, 1).view(head_num, -1, model_dim)
        multi_k = k.repeat(head_num, 1, 1).view(head_num, -1, model_dim)
        multi_v = v.repeat(head_num, 1, 1).view(head_num, -1, model_dim)

        multi_q = torch.bmm(multi_q, self.w_qs).view(-1, q_len, k_dim)
        multi_k = torch.bmm(multi_k, self.w_ks).view(-1, k_len, k_dim)
        multi_v = torch.bmm(multi_v, self.w_vs).view(-1, v_len, v_dim)

        attention_output = self.attention_layer(multi_q, multi_k, multi_v,
                                                mask=attention_mask.repeat(head_num, 1, 1))

        tmp_output = torch.cat(torch.split(attention_output, batch_size, dim=0), dim=-1)

        linear_output = self.linear_layer(tmp_output)

        dropout_output = self.dropout_layer(linear_output)

        return self.norm_layer(dropout_output + res)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, hid_dim, inner_hid_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(hid_dim, inner_hid_dim, 1) # position-wise
        self.w_2 = nn.Conv1d(inner_hid_dim, hid_dim, 1) # position-wise
        self.layer_norm = LayerNormalization(hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)

# Layer
class EncoderLayer(nn.Module):

    def __init__(self, model_dim, inner_hid_dim, head_num, k_dim, v_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention_layer = MultiHeadAttentionUnit(
            head_num=head_num,
            model_dim=model_dim,
            k_dim=k_dim,
            v_dim=v_dim,
            dropout=dropout
        )

        self.feedforward_layer = PositionwiseFeedForward(
            hid_dim=model_dim,
            inner_hid_dim=inner_hid_dim,
            dropout=dropout
        )

    def forward(self, input, attention_mask=None):
        attention_output = self.attention_layer(
            q=input,
            k=input,
            v=input,
            attention_mask=attention_mask
        )
        feedforward_output = self.feedforward_layer(attention_output)
        return feedforward_output

# Model
def position_encoding_init(position_num, pos_vec_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / pos_vec_dim) for j in range(pos_vec_dim)]
        if pos != 0 else np.zeros(pos_vec_dim) for pos in range(position_num)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_attention_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

class Encoder(nn.Module):

    def __init__(self, vocab_size, sentence_length, layer_num, head_num, k_dim, v_dim,
                 word_vec_dim, model_dim, inner_hid_dim, word_vec_matrix, dropout=0.1):
        super(Encoder, self).__init__()

        position_num = sentence_length + 1
        self.sentence_length = sentence_length
        self.model_dim = model_dim

        self.position_embedding = nn.Embedding(position_num, word_vec_dim, padding_idx=0)
        self.position_embedding.weight.data = position_encoding_init(position_num, word_vec_dim)

        self.word_embedding = nn.Embedding(vocab_size, word_vec_dim, padding_idx=0)
        self.word_embedding.weight.data = torch.from_numpy(word_vec_matrix)

        self.layer_stack = nn.ModuleList(
            [EncoderLayer(model_dim=model_dim,
                          inner_hid_dim=inner_hid_dim,
                          head_num=head_num,
                          k_dim=k_dim,
                          v_dim=v_dim,
                          dropout=dropout)
             for _ in range(layer_num)
             ]
        )

    def forward(self, word_seq, pos_seq):
        word_embedding_output = self.word_embedding(word_seq)
        position_embedding_output = self.position_embedding(pos_seq)

        embedding_output = word_embedding_output.float() + position_embedding_output.float()

        output = embedding_output
        attention_mask = get_attention_padding_mask(word_seq, word_seq)

        for encoder_layer in self.layer_stack:
            output = encoder_layer(output, attention_mask)

        return output

class Encoder_Max_Pooling(nn.Module):

    def __init__(self, vocab_size, sentence_length, layer_num, head_num, k_dim, v_dim,
                 word_vec_dim, model_dim, inner_hid_dim, word_vec_matrix, dropout=0.1):

        super(Encoder_Max_Pooling, self).__init__()

        self.Encoder = Encoder(vocab_size, sentence_length, layer_num, head_num, k_dim, v_dim,
                               word_vec_dim, model_dim, inner_hid_dim, word_vec_matrix, dropout)

    def forward(self, word_seq, pos_seq):
        output = self.Encoder(word_seq, pos_seq)
        max_pooling_output = torch.max(output, 1)[0]
        return max_pooling_output

class Sent_Encoder_Max_Pooling(nn.Module):
    def __init__(self, paragraph_length, layer_num, head_num, k_dim, v_dim,
                 input_vec_dim, model_dim, inner_hid_dim, dropout=0.1):
        super(Sent_Encoder_Max_Pooling, self).__init__()

        position_num = paragraph_length + 1
        self.sentence_length = paragraph_length
        self.model_dim = model_dim

        self.position_embedding = nn.Embedding(position_num, input_vec_dim, padding_idx=0)
        self.position_embedding.weight.data = position_encoding_init(position_num, input_vec_dim)

        self.layer_stack = nn.ModuleList(
            [EncoderLayer(model_dim=model_dim,
                          inner_hid_dim=inner_hid_dim,
                          head_num=head_num,
                          k_dim=k_dim,
                          v_dim=v_dim,
                          dropout=dropout)
             for _ in range(layer_num)
             ]
        )

    def forward(self, para_matrix, pos_seq):
        position_embedding_output = self.position_embedding(pos_seq)

        embedding_output = para_matrix.float() + position_embedding_output.float()

        output = embedding_output

        for encoder_layer in self.layer_stack:
            output = encoder_layer(output)

        max_pooling_output = torch.max(output, 1)[0]

        return max_pooling_output


class TransformerEncoder_BiLSTM(nn.Module):

    def __init__(self, encoder_vocab_size, encoder_sentence_length, encoder_layer_num, encoder_head_num, encoder_k_dim,
                 encoder_v_dim, encoder_word_vec_dim, encoder_model_dim, encoder_inner_hid_dim, word_vec_matrix,
                 sent_hidden_dim, sent_fc_dim, sent_dropout, tagset_size):

        super(TransformerEncoder_BiLSTM, self).__init__()

        self.sentence_encoder = Encoder_Max_Pooling(vocab_size=encoder_vocab_size,
                                                    sentence_length=encoder_sentence_length,
                                                    layer_num=encoder_layer_num,
                                                    head_num=encoder_head_num,
                                                    k_dim=encoder_k_dim,
                                                    v_dim=encoder_v_dim,
                                                    word_vec_dim=encoder_word_vec_dim,
                                                    model_dim=encoder_model_dim,
                                                    inner_hid_dim=encoder_inner_hid_dim,
                                                    word_vec_matrix=word_vec_matrix)

        self.sent_lstm = nn.LSTM(input_size=encoder_model_dim, hidden_size=sent_hidden_dim, bidirectional=True, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(2 * sent_hidden_dim, sent_fc_dim),
            nn.Dropout(sent_dropout),
            nn.ReLU(),
            nn.Linear(sent_fc_dim, sent_fc_dim),
            nn.Dropout(sent_dropout),
            nn.ReLU(),
            nn.Linear(sent_fc_dim, tagset_size)
        )

    def forward(self, word_seq, pos_seq):
        sentence_encoder_out = self.sentence_encoder(word_seq, pos_seq)
        sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(len(sentence_encoder_out), 1, -1))
        tag_space = self.classifier(sent_lstm_out.view(len(sent_lstm_out), -1))
        return tag_space

class BiLSTM_TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, fc_dim, vocab_size, tagset_size, word_vec_matrix, dropout,
                 trans_sentence_length, trans_layer_num, trans_head_num, trans_k_dim, trans_v_dim,
                 trans_model_dim, trans_inner_hid_dim):
        super(BiLSTM_TransformerEncoder, self).__init__()

        self.sentence_encoder = BiLstmSentenceEncoder(embedding_dim=embedding_dim,
                                                      hidden_dim=hidden_dim,
                                                      vocab_size=vocab_size,
                                                      word_vec_matrix=word_vec_matrix)

        self.sent_transformer = Encoder(vocab_size=vocab_size,
                                        sentence_length=trans_sentence_length,
                                        layer_num=trans_layer_num,
                                        head_num=trans_head_num,
                                        k_dim=trans_k_dim,
                                        v_dim=trans_v_dim,
                                        word_vec_matrix=word_vec_matrix,
                                        model_dim=trans_model_dim,
                                        inner_hid_dim=trans_inner_hid_dim,
                                        word_vec_dim=embedding_dim
                                        )

        self.sent_lstm = nn.LSTM(input_size=2*embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, tagset_size)
        )

    def forward(self, sentence_tuple):
        sentence_encoder_out = self.sentence_encoder(sentence_tuple)
        sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(len(sentence_encoder_out), 1, -1))
        tag_space = self.classifier(sent_lstm_out.view(len(sent_lstm_out), -1))
        return tag_space
















