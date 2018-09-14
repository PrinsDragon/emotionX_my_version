import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from Attention_Net import ScaledDotProductAttention_Batch

# GPU = True
#
# def mask_initial(sentence_length_list):
#     batch_size = len(sentence_length_list)
#     max_length = int(max(sentence_length_list))
#     mask = torch.zeros(batch_size, max_length, max_length).byte()
#     for i in range(batch_size):
#         sent_len = int(sentence_length_list[i])
#         len_mask = np.ones((max_length, 1))
#         len_mask[sent_len:] = 0
#         attn_mask = np.matmul(len_mask, len_mask.transpose())
#         attn_mask = torch.from_numpy(attn_mask)
#         attn_mask = torch.eq(attn_mask, 0)
#         mask[i] = attn_mask
#     if GPU:
#         mask = mask.cuda()
#     return mask

GPU = True

class BiLSTM_Attention_Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word_vec_matrix):
        super(BiLSTM_Attention_Encoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_vec_matrix))

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

        self.attention_layer = ScaledDotProductAttention_Batch(model_dim=2*embedding_dim)

    def forward(self, sentence_tuple):
        # split input
        sentence = sentence_tuple[0]
        sentence_length_list = sentence_tuple[1]

        # eliminate extra zeros
        max_len = int(sentence_length_list.max())
        sentence = sentence[:, 0:max_len]

        # get word embedding
        embeds = self.word_embeddings(sentence)

        # sort
        sentence_length_list, indices = torch.sort(sentence_length_list, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)

        embeds = embeds[indices]

        embeds = pack_padded_sequence(embeds, sentence_length_list.cpu().numpy(), batch_first=True)
        lstm_out, _ = self.lstm(embeds)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # unsort
        lstm_out = lstm_out[desorted_indices]

        attention_out = self.attention_layer(lstm_out, lstm_out, lstm_out)

        # res_plus = attention_out + lstm_out

        # max_pooling_out = torch.max(res_plus, 1)[0]
        max_pooling_out = torch.max(attention_out, 1)[0]

        return max_pooling_out, attention_out

class Whole_Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word_vec_matrix):
        super(Whole_Encoder, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_vec_matrix))

        self.whole_bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

        self.attention_layer = ScaledDotProductAttention_Batch(model_dim=2*embedding_dim)

    def forward(self, sentence_tuple):
        # split input
        sentence = sentence_tuple[0]
        sentence_length_list = sentence_tuple[1]

        # sentence_max_length = int(sentence_length_list.max())
        sentence_num = len(sentence)

        # cat
        sentence_whole = sentence[0][:sentence_length_list[0]]
        for i in range(1, sentence_num):
            sent = sentence[i]
            sent_len = sentence_length_list[i]
            sentence_whole = torch.cat([sentence_whole, sent[:sent_len]])

        sentence_length_sum = copy.deepcopy(sentence_length_list)
        for i in range(1, sentence_num):
            sentence_length_sum[i] += sentence_length_sum[i-1]

        # get word embedding
        embeds = self.word_embeddings(sentence_whole)

        # embedding
        lstm_out, _ = self.whole_bilstm(embeds.view(1, embeds.shape[0], -1))
        lstm_out = lstm_out.view(lstm_out.shape[1], -1)

        # split

        # encoder_out = lstm_out[sentence_length_list[0]/2].view(1, -1)
        #
        # for i in range(1, sentence_num):
        #     index = sentence_length_sum[i-1] + sentence_length_list[i]/2
        #     encoder_out = torch.cat([encoder_out, lstm_out[index].view(1, -1)], 0)

        # encoder_out = torch.max(lstm_out[:sentence_length_list[0]], 0)[0].view(1, -1)
        # for i in range(1, sentence_num):
        #     start = sentence_length_sum[i-1]
        #     end = sentence_length_sum[i]
        #     sent = lstm_out[start:end]
        #     cat = torch.max(sent, 0)[0].view(1, -1)
        #     encoder_out = torch.cat([encoder_out, cat], 0)

        encoder_out = lstm_out[:sentence_length_list[0]]
        encoder_out = encoder_out.view(1, encoder_out.shape[0], -1)

        encoder_out = self.attention_layer(encoder_out, encoder_out, encoder_out)
        encoder_out = torch.max(encoder_out, 1)[0]

        for i in range(1, sentence_num):
            start = sentence_length_sum[i-1]
            end = sentence_length_sum[i]
            sent = lstm_out[start:end]
            sent = sent.view(1, sent.shape[0], -1)

            sent = self.attention_layer(sent, sent, sent)
            sent = torch.max(sent, 1)[0]

            encoder_out = torch.cat([encoder_out, sent])

        return encoder_out, _


def make_cat_matrix(a, b):
    abs_part = torch.abs(a - b)
    multiply_part = a * b
    cat = torch.cat([a, b, abs_part, multiply_part], 2)
    return cat

class Attention_Projector_BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, fc_dim, tagset_size, dropout):
        super(Attention_Projector_BiLSTM, self).__init__()

        self.attention_layer = ScaledDotProductAttention_Batch(model_dim=2*embedding_dim)

        self.first_projector = nn.Sequential(
            nn.Linear(2 * embedding_dim * 4, embedding_dim),
            nn.ReLU()
        )
        self.second_projector = nn.Sequential(
            nn.Linear(2 * embedding_dim * 4, embedding_dim),
            nn.ReLU()
        )

        self.first_bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.second_bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

        self.final_projector = nn.Sequential(
            nn.Linear(2 * embedding_dim * 4, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, tagset_size)
        )

    def forward(self, first_matrix, second_matrix):
        first_attention_out = self.attention_layer(first_matrix, second_matrix, second_matrix)
        second_attention_out = self.attention_layer(second_matrix, first_matrix, first_matrix)

        first_cat = make_cat_matrix(first_matrix, first_attention_out)
        second_cat = make_cat_matrix(second_matrix, second_attention_out)

        first_projector_out = self.first_projector(first_cat)
        second_projector_out = self.second_projector(second_cat)

        first_bilstm_out, _ = self.first_bilstm(first_projector_out)
        second_bilstm_out, _ = self.second_bilstm(second_projector_out)

        first_max_pooling_out = torch.max(first_bilstm_out, 1)[0]
        second_max_pooling_out = torch.max(second_bilstm_out, 1)[0]

        abs_part = torch.abs(first_max_pooling_out - second_max_pooling_out)
        multiply_part = first_max_pooling_out * second_max_pooling_out
        cat = torch.cat([first_max_pooling_out, second_max_pooling_out, abs_part, multiply_part], 1)

        tag_space = self.final_projector(cat)

        zero = torch.zeros(1, 8)
        if GPU:
            zero = zero.cuda()
        tag_space = torch.cat([zero, tag_space])

        return tag_space

class BiLSTM_Atention_BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, fc_dim, vocab_size, tagset_size, word_vec_matrix, dropout):
        super(BiLSTM_Atention_BiLSTM, self).__init__()

        # self.sentence_encoder = BiLSTM_Attention_Encoder(embedding_dim=embedding_dim,
        #                                                  hidden_dim=hidden_dim,
        #                                                  vocab_size=vocab_size,
        #                                                  word_vec_matrix=word_vec_matrix)

        self.sentence_encoder = Whole_Encoder(embedding_dim=embedding_dim,
                                              hidden_dim=hidden_dim,
                                              vocab_size=vocab_size,
                                              word_vec_matrix=word_vec_matrix)

        self.sent_lstm = nn.LSTM(input_size=2*embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

        # self.attention_layer = ScaledDotProductAttention_Batch(model_dim=2*embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, tagset_size)
        )

        self.qa_score_linear = nn.Sequential(
            nn.Linear(2*embedding_dim*4, 100),
            nn.Linear(100, 1)
        )

        # self.attention_projector_bilstm = Attention_Projector_BiLSTM(embedding_dim, hidden_dim, fc_dim, tagset_size, dropout)

    def forward(self, sentence_tuple):
        sentence_encoder_out, sentence_encoder_matrix = self.sentence_encoder(sentence_tuple)
        sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(1, sentence_encoder_out.shape[0], -1))

        # attention_out = self.attention_layer(sent_lstm_out, sent_lstm_out, sent_lstm_out)
        # tag_space = self.classifier(attention_out.view(attention_out.shape[1], -1))

        tag_space = self.classifier(sent_lstm_out.view(sent_lstm_out.shape[1], -1))

        # tag_space = self.classifier(sentence_encoder_out)

        # extra_tag_space = self.attention_projector_bilstm(sentence_encoder_matrix[:-1],
        #                                                   sentence_encoder_matrix[1:])

        # return sentence_encoder_out, tag_space + extra_tag_space

        return sentence_encoder_out, tag_space

    def question_answer_score(self, question_tensor, answer_tensor):
        abs_part = torch.abs(question_tensor - answer_tensor)
        multiply_part = question_tensor * answer_tensor
        cat_tensor = torch.cat([question_tensor, answer_tensor, abs_part, multiply_part], 0)
        score = self.qa_score_linear(cat_tensor)
        return score
        # question_matrix = question_tensor.view(1, -1)
        # answer_matrix = answer_tensor.view(1, -1)
        #
        # return torch.mm(question_matrix, answer_matrix.t()).view([])

    # multitask loss
    def get_loss(self, sentence_encoder_out, tag_space, emotion_loss_func, targets):
        # sentence_encoder_out, _ = self.sentence_encoder(sentence_tuple)
        # sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(1, sentence_encoder_out.shape[0], -1))
        #
        # # attention_out = self.attention_layer(sent_lstm_out, sent_lstm_out, sent_lstm_out)
        # # tag_space = self.classifier(attention_out.view(attention_out.shape[1], -1))
        #
        # tag_space = self.classifier(sent_lstm_out.view(sent_lstm_out.shape[1], -1))
        #
        # # tag_space = self.classifier(sentence_encoder_out)

        emotion_loss = emotion_loss_func(tag_space, targets)

        # return emotion_loss, emotion_loss, 0

        # self.loss = tf.reduce_mean(tf.nn.relu(1 + self.qa_score_1 - self.qa_score_2)
        sentence_num = len(sentence_encoder_out)

        batch_loss = []
        for i in range(sentence_num - 1):
            sent_loss = []

            sent = sentence_encoder_out[i]
            next_sent = sentence_encoder_out[i+1]
            true_score = self.question_answer_score(sent, next_sent)

            # rand_sample = random.sample([i for i in range(sentence_num)], 10)

            for j in range(sentence_num):
                if j != i and j != i+1:
                    false_sent = sentence_encoder_out[j]
                    false_score = self.question_answer_score(sent, false_sent)
                    sent_loss.append(F.relu(1 + true_score - false_score))

            sent_loss_sum = sum(sent_loss)
            sent_loss_mean = sent_loss_sum / len(sent_loss)
            batch_loss.append(sent_loss_mean)

        batch_loss_sum = sum(batch_loss)
        batch_loss_mean = batch_loss_sum / len(batch_loss)

        loss = emotion_loss + batch_loss_mean
        # print("emotion loss: {:.3f} answer loss: {:.3f}".format(float(emotion_loss), float(batch_loss_mean)))

        # loss = batch_loss_mean
        # print("answer loss: {:.6f}".format(float(loss)))

        return loss, emotion_loss, batch_loss_mean
