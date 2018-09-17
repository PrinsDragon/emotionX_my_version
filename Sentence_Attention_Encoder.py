import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from Attention_Net import ScaledDotProductAttention_Batch

GPU = True

class Three_Attention_Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word_vec_matrix):
        super(Three_Attention_Encoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_vec_matrix))

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

        self.attention_layer = ScaledDotProductAttention_Batch(model_dim=2*embedding_dim)

        # self.projector = nn.Linear(2*embedding_dim*3, 2*embedding_dim)

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

        self_attention_out = self.attention_layer(lstm_out, lstm_out, lstm_out)

        before_part = lstm_out[:-1]
        after_part = lstm_out[1:]

        before_attention_out = self.attention_layer(after_part, before_part, before_part)
        after_attention_out = self.attention_layer(before_part, after_part, after_part)

        size_a, size_b = lstm_out.shape[1:]
        zero_append = torch.zeros(1, size_a, size_b)
        if GPU:
            zero_append = zero_append.cuda()

        before_attention_out = torch.cat([before_attention_out, zero_append])
        after_attention_out = torch.cat([zero_append, after_attention_out])

        attention_cat = torch.cat([before_attention_out, self_attention_out, after_attention_out], 2)

        # projector_out = self.projector(attention_cat)

        max_pooling_out = torch.max(attention_cat, 1)[0]

        return max_pooling_out

class BiLSTM_Atention_BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, fc_dim, vocab_size, tagset_size, word_vec_matrix, dropout):
        super(BiLSTM_Atention_BiLSTM, self).__init__()

        self.sentence_encoder = Three_Attention_Encoder(embedding_dim=embedding_dim,
                                                        hidden_dim=hidden_dim,
                                                        vocab_size=vocab_size,
                                                        word_vec_matrix=word_vec_matrix)

        sentence_encoder_dim = 2*embedding_dim*3

        self.sent_lstm = nn.LSTM(input_size=sentence_encoder_dim, hidden_size=hidden_dim*3, bidirectional=True, batch_first=True)

        # self.attention_layer = ScaledDotProductAttention_Batch(model_dim=2*embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim * 3, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, tagset_size)
        )

        self.qa_score_linear = nn.Sequential(
            nn.Linear(sentence_encoder_dim*4, 100),
            nn.Linear(100, 1)
        )

    def forward(self, sentence_tuple):
        sentence_encoder_out = self.sentence_encoder(sentence_tuple)
        sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(1, sentence_encoder_out.shape[0], -1))

        # attention_out = self.attention_layer(sent_lstm_out, sent_lstm_out, sent_lstm_out)
        # tag_space = self.classifier(attention_out.view(attention_out.shape[1], -1))

        tag_space = self.classifier(sent_lstm_out.view(sent_lstm_out.shape[1], -1))

        # tag_space = self.classifier(sentence_encoder_out)

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
        # sentence_encoder_out = self.sentence_encoder(sentence_tuple)
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
