#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np
import json

print("GPU available: ", torch.cuda.is_available())

# parameters
mode = 4
print("Now running mode = ", mode)

train_dir = "./data/Friends_Proc_{}/friends_seq_train.json".format(mode)
dev_dir = "./data/Friends_Proc_{}/friends_seq_dev.json".format(mode)
test_dir = "./data/Friends_Proc_{}/friends_seq_test.json".format(mode)
word_vector_dir = "./data/Friends_Proc_{}/friends_word_vec.txt".format(mode)

echo_num = 50
embedding_dim = 300
hidden_dim = 300
fc_dim = 128
batch_size = 128
gradient_max_norm = 5
target_size = mode
dropout_rate = 0.5

# Data

def build_word_vec_matrix(word_vec_dir):
    print("Start to Build Word Vec Matrix ...")

    word_vec_matrix = ""
    word_num = 0
    word_in_vec = 0
    word_vec_file = open(word_vec_dir, "r", encoding="utf-8")
    for i, line in enumerate(word_vec_file):
        if i == 0:
            word_num = int(line)+1
            word_vec_matrix = np.zeros((word_num, embedding_dim))
            continue
        try:
            id, vec = line.split(' ', 1)
            word_vec_matrix[int(id)] = np.array(list(map(float, vec.split())))
            word_in_vec += 1
            if word_in_vec % 500 == 0:
                print("{} words finish ...".format(word_in_vec))
        except:
            continue

    print("Total {} words in vec".format(word_in_vec))
    print("Finish !")
    return word_num, word_vec_matrix

class Sentence:
    def __init__(self, name, seq, label):
        self.name = name
        self.seq = seq
        self.seq_len = len(seq)
        self.label = label

    def extend(self, total_length):
        tmp = torch.zeros(total_length-self.seq_len).long()
        self.seq = torch.cat([self.seq, tmp], 0)
        # print(self.text_in_seq)

    # def extend(self, total_length):
    #     extend_times = total_length // self.text_len
    #     self.text = self.text * extend_times
    #     tmp = [0 for _ in range(0, total_length - self.text_len)]
    #     self.text = self.text + tmp
    #
    # def cat_len(self):
    #     self.seq[-1] = self.seq_len
    #
    # def sentence2feed(self, word_vector_dictionary, max_sentence_length):
    #     embedding_size = 300
    #     text_in_number = np.zeros([max_sentence_length, embedding_size])
    #
    #     for k in range(0, len(self.text)):
    #         if self.text[k] in word_vector_dictionary.wv:
    #             text_in_number[k] = word_vector_dictionary[self.text[k]]
    #
    #     text_length = len(self.text)
    #     ex = max_sentence_length // text_length
    #
    #     for k in range(1, ex):
    #         text_in_number[text_length*k : text_length*(k+1)] = text_in_number[0 : text_length]
    #
    #     return text_in_number, self.label

class EmotionDataSet(Dataset):
    def __init__(self, data_dir):
        data_file = open(data_dir)
        data = json.load(data_file)

        self.sentences = []
        self.max_sentence_length = 0
        self.emotion_num = {i : 0 for i in range(target_size)}

        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                name = data[i][j]["speaker"]
                seq = torch.LongTensor(list(map(int, data[i][j]["utterance"].split())))
                # text = data[i][j]["utterance"].split(" ")
                label = int(data[i][j]["emotion"])

                self.emotion_num[label] += 1
                self.max_sentence_length = max(self.max_sentence_length, len(seq))
                self.sentences.append(Sentence(name=name, seq=seq, label=label))

        for sent in self.sentences:
            sent.extend(self.max_sentence_length)
            # sent.cat_len()

        self.sentences_num = self.sentences.__len__()

    def __getitem__(self, index):
        return self.sentences[index].seq, self.sentences[index].seq_len, self.sentences[index].label

    def __len__(self):
        return self.sentences_num

# Load

train_dataset = EmotionDataSet(data_dir=train_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

dev_dataset = EmotionDataSet(data_dir=dev_dir)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

test_dataset = EmotionDataSet(data_dir=test_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# Net

START = target_size
STOP = target_size + 1

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # self.tag_to_ix = tag_to_ix
        self.tagset_size = tagset_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[START, :] = -10000
        self.transitions.data[:, STOP] = -10000

        # self.hidden = self.init_hidden()

    # def init_hidden(self):
    #     return (torch.randn(2, 1, self.hidden_dim // 2),
    #             torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][START] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[STOP]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence) # .view(len(sentence), 1, -1)
        lstm_out, _ = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([START], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[STOP, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][START] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == START  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

class myLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, fc_dim, batch_size, vocab_size, tagset_size, max_sentence_length, word_vec_matrix, dropout):
        super(myLSTM, self).__init__()

        self.max_sentence_length = max_sentence_length
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_vec_matrix))

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.sent_lstm = nn.LSTM(input_size=2*embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, tagset_size)
        )
        # self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, batch_size, self.hidden_dim),
                torch.zeros(2, batch_size, self.hidden_dim))

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

        sent_lstm_out, _ = self.sent_lstm(max_pooling_out.view(len(max_pooling_out), 1, -1))

        tag_space = self.classifier(sent_lstm_out.view(len(sent_lstm_out), -1))

        # tag_space = F.sigmoid(tag_space)

        # tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_space

        # return tag_scores

class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, word_vec_matrix):
        super(SentenceEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_vec_matrix))

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

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

        self.sentence_encoder = SentenceEncoder(embedding_dim=embedding_dim,
                                                hidden_dim=hidden_dim,
                                                vocab_size=vocab_size,
                                                tagset_size=tagset_size,
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
        sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(len(sentence_encoder_out), 1, -1))
        tag_space = self.classifier(sent_lstm_out.view(len(sent_lstm_out), -1))
        return tag_space

class BiLSTM_BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, fc_dim, vocab_size, tagset_size, word_vec_matrix, dropout):
        super(BiLSTM_BiLSTM_CRF, self).__init__()

        self.sentence_encoder = SentenceEncoder(embedding_dim=embedding_dim,
                                                hidden_dim=hidden_dim,
                                                vocab_size=vocab_size,
                                                tagset_size=tagset_size,
                                                word_vec_matrix=word_vec_matrix)

        self.BiLSTM_CRF = BiLSTM_CRF(vocab_size=vocab_size,
                                     tagset_size=tagset_size,
                                     embedding_dim=embedding_dim,
                                     hidden_dim=hidden_dim
                                     )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, tagset_size)
        )

    def loss(self):
        123

    def forward(self, sentence_tuple):
        sentence_encoder_out = self.sentence_encoder(sentence_tuple)
        sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(len(sentence_encoder_out), 1, -1))
        tag_space = self.classifier(sent_lstm_out.view(len(sent_lstm_out), -1))
        return tag_space

def print_info(sign, total_loss, total_acc, acc, dataset):
    print("{}: Loss: {:.6f}, Acc: {:.6f}".format(sign, total_loss / (len(dataset)), total_acc.float() / (len(dataset))))

    eval = [acc[i] / dataset.emotion_num[i] for i in range(target_size)]

    if target_size == 4:
        print("{}: 0: {:.6f}, 1: {:.6f}, 2: {:.6f}, 3: {:.6f}".format(sign, eval[0], eval[1], eval[2], eval[3]))
    elif target_size == 8:
        print("{}: 0: {:.6f}, 1: {:.6f}, 2: {:.6f}, 3: {:.6f}, 4: {:.6f}, 5: {:.6f}, 6: {:.6f}, 7: {:.6f}".format(
            sign,
            eval[0],
            eval[1],
            eval[2],
            eval[3],
            eval[4],
            eval[5],
            eval[6],
            eval[7]
        ))

    average_acc = sum(eval) / target_size
    print("{}: Average Acc: {:.6f}\n".format(sign, average_acc))

    return average_acc

vocab_size, word_vec_matrix = build_word_vec_matrix(word_vector_dir)

model = myLSTM(embedding_dim=embedding_dim,
               hidden_dim=hidden_dim,
               fc_dim=fc_dim,
               batch_size=batch_size,
               vocab_size=vocab_size,
               tagset_size=target_size,
               max_sentence_length=train_dataset.max_sentence_length,
               word_vec_matrix=word_vec_matrix,
               dropout=dropout_rate)

print(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

# train & dev
# region train & dev
print("**********************************")
print("train_dataset: ", train_dataset.emotion_num)
print("dev_dataset: ", dev_dataset.emotion_num)

max_dev_average_acc = 0
max_dev_average_acc_model_state = ""

for epoch in range(echo_num):
    # train
    model.train()

    print("----------------------------------")
    print('epoch {}'.format(epoch + 1))
    train_acc = {i: 0. for i in range(target_size)}
    total_acc = 0.
    train_loss = 0.
    for batch_times, (batch_x, batch_x_len, batch_y) in enumerate(train_loader):
        if batch_times % 20 == 0:
            print("Sentences : ", batch_times * batch_size)

        sentence_in = (batch_x, batch_x_len)
        targets = batch_y

        tag_scores = model(sentence_in)

        pred = torch.max(tag_scores, 1)[1]

        total_acc += (pred == targets).sum()

        for i in range(len(pred)):
            if pred[i] == targets[i]:
                train_acc[int(targets[i])] += 1

        loss = loss_func(tag_scores, targets)

        train_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        shrink_factor = 1
        total_norm = 0

        for p in model.parameters():
            if p.requires_grad:
                try:
                    p.grad.data.div_(batch_size)
                    total_norm += p.grad.data.norm() ** 2
                except:
                    pass
        total_norm = np.sqrt(total_norm)

        if total_norm > gradient_max_norm:
            print("gradient clipping")
            shrink_factor = gradient_max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

    print_info(sign="Train", total_loss=train_loss, total_acc=total_acc, acc=train_acc, dataset=train_dataset)

    # dev
    # region dev
    model.eval()

    dev_acc = {i: 0. for i in range(target_size)}
    total_acc = 0.
    dev_loss = 0.
    for batch_x, batch_x_len, batch_y in dev_loader:

        sentence_in = (batch_x, batch_x_len)
        targets = batch_y

        # if sentence_in.shape[0] != batch_size:
        #     continue

        tag_scores = model(sentence_in)

        pred = torch.max(tag_scores, 1)[1]

        total_acc += (pred == targets).sum()

        for i in range(len(pred)):
            if pred[i] == targets[i]:
                dev_acc[int(targets[i])] += 1

        loss = loss_func(tag_scores, targets)

        dev_loss += loss

    dev_average_acc = print_info(sign="Dev", total_loss=dev_loss, total_acc=total_acc, acc=dev_acc, dataset=dev_dataset)

    if dev_average_acc > max_dev_average_acc:
        max_dev_average_acc = dev_average_acc
        max_dev_average_acc_model_state = model.state_dict()
        print("### new max dev acc !\n")
    else:
        print("Dev: Now Max Acc: {:.6f}\n".format(max_dev_average_acc))

    # endregion

print("echo = {} max dev acc = {:.6f}\n".format(echo_num, max_dev_average_acc))

# endregion

# test
# region test
print("**********************************")
print("test_dataset: ", test_dataset.emotion_num)

# load max dev state
model.load_state_dict(max_dev_average_acc_model_state)

test_acc = {i : 0. for i in range(target_size)}
total_acc = 0.
test_loss = 0.
for batch_x, batch_x_len, batch_y in test_loader:

    sentence_in = (batch_x, batch_x_len)
    targets = batch_y

    tag_scores = model(sentence_in)

    pred = torch.max(tag_scores, 1)[1]

    total_acc += (pred == targets).sum()

    for i in range(len(pred)):
        if pred[i] == targets[i]:
            test_acc[int(targets[i])] += 1

    loss = loss_func(tag_scores, targets)

    test_loss += loss

print_info(sign="Test", total_loss=test_loss, total_acc=total_acc, acc=test_acc, dataset=test_dataset)

# endregion
