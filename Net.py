import torch
import torch.nn as nn
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

    def forward(self, sentence_tuple):
        sentence_encoder_out = self.sentence_encoder(sentence_tuple)
        sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(1, sentence_encoder_out.shape[0], -1))
        tag_space = self.classifier(sent_lstm_out.view(sent_lstm_out.shape[1], -1))
        return tag_space
















