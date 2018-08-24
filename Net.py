import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# Net
class BiLstmSentenceEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word_vec_matrix):
        super(BiLstmSentenceEncoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_vec_matrix))

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

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

        # max_pooling
        max_pooling_out = torch.max(lstm_out, 1)[0]

        #  don't know why in emotionX: the axis = 1 ?????
        # max_pooling_out = F.max_pool1d(lstm_out, kernel_size=600)

        return max_pooling_out

class BiLSTM_BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, fc_dim, vocab_size, tagset_size, word_vec_matrix, dropout):
        super(BiLSTM_BiLSTM, self).__init__()

        self.sentence_encoder = BiLstmSentenceEncoder(embedding_dim=embedding_dim,
                                                      hidden_dim=hidden_dim,
                                                      vocab_size=vocab_size,
                                                      word_vec_matrix=word_vec_matrix)

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

        self.qa_score_linear = nn.Sequential(
            nn.Linear(2*embedding_dim*4, 100),
            nn.Linear(100, 1)
        )

    def forward(self, sentence_tuple):
        sentence_encoder_out = self.sentence_encoder(sentence_tuple)
        sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(len(sentence_encoder_out), 1, -1))
        tag_space = self.classifier(sent_lstm_out.view(len(sent_lstm_out), -1))
        return tag_space

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
    def get_loss(self, sentence_tuple, emotion_loss_func, targets):
        sentence_encoder_out = self.sentence_encoder(sentence_tuple)

        sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(len(sentence_encoder_out), 1, -1))
        tag_space = self.classifier(sent_lstm_out.view(len(sent_lstm_out), -1))
        emotion_loss = emotion_loss_func(tag_space, targets)

        # return emotion_loss

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

        # loss = emotion_loss + batch_loss_mean

        # print("emotion loss: {:.3f} answer loss: {:.3f}".format(float(emotion_loss), float(batch_loss_mean)))

        loss = batch_loss_mean
        print("answer loss: {}".format(float(loss)))

        return loss
