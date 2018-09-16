#coding=utf-8

import json
import random
import time
import os
import copy
import sys

from data.word_id_helper import read_word_id, ori_sentence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from Net import BiLSTM_BiLSTM
# from BiLSTM_Attention_CRF import BiLSTM_Atention_BiLSTM
# from QA_Attention_Net import BiLSTM_Atention_BiLSTM
from Sentence_Attention_Encoder import BiLSTM_Atention_BiLSTM

# from Attention_Net import TransformerEncoder_BiLSTM
# from Attention_Net import BiLSTM_TransformerEncoder
# from Attention_Net import BiLSTM_Attention

GPU = torch.cuda.is_available()
SERVER = True

# parameters
mode = 4

epoch_num = 100
embedding_dim = 300
hidden_dim = 300
fc_dim = 128
batch_size = 128
gradient_max_norm = 5
target_size = 8
dropout_rate = 0.8

if SERVER:
    TAG = "epoc={}_{}".format(epoch_num, "Bilstm+3Attention+bilstm+qa")
    TIME = time.strftime('%Y.%m.%d-%H:%M', time.localtime(time.time()))

    save_dir = "./checkpoints/{}_checkpoint_{}/".format(TIME, TAG)

    try:
        os.makedirs(save_dir)
    except:
        pass

    save_mistake_sent = open(save_dir + "mistake_sent_{}.vstxt".format(TAG), "w", encoding="utf-8")
    save_out = open(save_dir + "out_{}.vstxt".format(TAG), "w", encoding="utf-8")
    sys.stdout = save_out

    print("GPU available: ", GPU)
    print(TAG)
    print(TIME)

train_dir = "./data/Merge_Proc/merge_seq_train.json"

friends_dev_dir = "./data/Merge_Proc/merge_seq_friends_dev.json"
emotionpush_dev_dir = "./data/Merge_Proc/merge_seq_emotionpush_dev.json"

friends_test_dir = "./data/Merge_Proc/merge_seq_friends_test.json"
emotionpush_test_dir = "./data/Merge_Proc/merge_seq_emotionpush_test.json"

word_vector_dir = "./data/Merge_Proc/merge_word_vec.txt"
word_id_dir = "./data/Merge_Proc/merge_word_id.txt"

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
            word_vec_matrix = 0.5 * np.random.random_sample((word_num, embedding_dim)) - 0.25
            word_vec_matrix[0] = 0
            # np.zeros((word_num, embedding_dim))
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
    print("Finish!")
    return word_num, word_vec_matrix

class Sentence:
    def __init__(self, name, seq, label):
        self.name = name
        self.seq = seq
        self.seq_len = len(seq)
        self.label = label

        self.pos_seq = torch.tensor([i for i in range(1, self.seq_len+1)])

    def extend(self, total_length):
        tmp = torch.zeros(total_length-self.seq_len).long()
        self.seq = torch.cat([self.seq, tmp], 0)
        self.pos_seq = torch.cat([self.pos_seq, tmp], 0)

class EmotionDataSet(Dataset):
    def __init__(self, data_dir):
        data_file = open(data_dir)
        data = json.load(data_file)

        self.sentences = []
        self.paragraphs = []
        self.max_sentence_length = 0
        self.max_paragraph_length = 0
        self.emotion_num = {i : 0 for i in range(target_size)}

        for i in range(0, len(data)):
            para = []
            for j in range(0, len(data[i])):
                name = data[i][j]["speaker"]
                seq = torch.LongTensor(list(map(int, data[i][j]["utterance"].split())))
                label = int(data[i][j]["emotion"])

                self.emotion_num[label] += 1
                self.max_sentence_length = max(self.max_sentence_length, len(seq))
                self.sentences.append(Sentence(name=name, seq=seq, label=label))
                para.append(Sentence(name=name, seq=seq, label=label))

            self.max_paragraph_length = max(self.max_paragraph_length, len(para))
            self.paragraphs.append(para)

        # for sent in self.sentences:
        #     sent.extend(self.max_sentence_length)

        for para in self.paragraphs:
            for sent in para:
                sent.extend(self.max_sentence_length)

            # random.shuffle(para)

        self.paragraphs_num = len(self.paragraphs)
        self.sentences_num = len(self.sentences)

    def __getitem__(self, index):
        # return self.sentences[index].seq, \
        #        self.sentences[index].pos_seq, \
        #        self.sentences[index].label

        return self.sentences[index].seq, self.sentences[index].seq_len, self.sentences[index].label

        # return ([sent.seq for sent in self.paragraphs[index]],
        #         [sent.seq_len for sent in self.paragraphs[index]],
        #         [sent.label for sent in self.paragraphs[index]])

    def __len__(self):
        # return self.sentences_num
        return self.paragraphs_num

    def get_paragraph(self):
        for para in self.paragraphs:
            para_tensor = para[0].seq.view(1, -1)
            sentence_lengths = torch.tensor([sent.seq_len for sent in para])
            sentence_labels = torch.tensor([sent.label for sent in para])
            for i in range(1, len(para)):
                seq_tensor = para[i].seq.view(1, -1)
                para_tensor = torch.cat([para_tensor, seq_tensor], 0)

            yield para_tensor, sentence_lengths, sentence_labels

# Load
train_dataset = EmotionDataSet(data_dir=train_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

friends_dev_dataset = EmotionDataSet(data_dir=friends_dev_dir)
friends_dev_loader = DataLoader(dataset=friends_dev_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
emotionpush_dev_dataset = EmotionDataSet(data_dir=emotionpush_dev_dir)
emotionpush_dev_loader = DataLoader(dataset=emotionpush_dev_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

dev_dataset = [friends_dev_dataset, emotionpush_dev_dataset]
dev_loader = [friends_dev_loader, emotionpush_dev_loader]

friends_test_dataset = EmotionDataSet(data_dir=friends_test_dir)
friends_test_loader = DataLoader(dataset=friends_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
emotionpush_test_dataset = EmotionDataSet(data_dir=emotionpush_test_dir)
emotionpush_test_loader = DataLoader(dataset=emotionpush_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

test_dataset = [friends_test_dataset, emotionpush_test_dataset]
test_loader = [friends_test_loader, emotionpush_test_loader]

word_id_dict = read_word_id(word_id_dir)

vocab_size, word_vec_matrix = build_word_vec_matrix(word_vector_dir)

model = BiLSTM_Atention_BiLSTM(embedding_dim=embedding_dim,
                               hidden_dim=hidden_dim,
                               fc_dim=fc_dim,
                               vocab_size=vocab_size,
                               tagset_size=target_size,
                               word_vec_matrix=word_vec_matrix,
                               dropout=dropout_rate)

# model = TransformerEncoder_BiLSTM(encoder_vocab_size=vocab_size,
#                                   encoder_sentence_length=max(train_dataset.max_sentence_length,
#                                                               dev_dataset.max_sentence_length,
#                                                               test_dataset.max_sentence_length),
#                                   encoder_layer_num=6,
#                                   encoder_head_num=8,
#                                   encoder_k_dim=64,
#                                   encoder_v_dim=64,
#                                   encoder_word_vec_dim=300,
#                                   encoder_model_dim=300,
#                                   encoder_inner_hid_dim=1024,
#                                   word_vec_matrix=word_vec_matrix,
#                                   sent_hidden_dim=hidden_dim,
#                                   sent_fc_dim=fc_dim,
#                                   sent_dropout=dropout_rate,
#                                   tagset_size=target_size)

# (self, embedding_dim, hidden_dim, fc_dim, vocab_size, tagset_size, word_vec_matrix, dropout,
#                  paragraph_length, layer_num, head_num, k_dim, v_dim, input_vec_dim, model_dim, inner_hid_dim):

# model = BiLSTM_TransformerEncoder(embedding_dim=embedding_dim,
#                                   hidden_dim=hidden_dim,
#                                   fc_dim=fc_dim,
#                                   vocab_size=vocab_size,
#                                   tagset_size=target_size,
#                                   word_vec_matrix=word_vec_matrix,
#                                   dropout=dropout_rate,
# 
#                                   paragraph_length=batch_size,
#                                   layer_num=1,
#                                   head_num=8,
#                                   k_dim=64,
#                                   v_dim=64,
#                                   input_vec_dim=2*embedding_dim,
#                                   model_dim=600,
#                                   inner_hid_dim=1024)

# self, embedding_dim, hidden_dim, fc_dim, vocab_size, tagset_size, word_vec_matrix, dropout,
#                  model_dim, max_paragraph_len):

# model = BiLSTM_Attention(embedding_dim=embedding_dim,
#                          hidden_dim=hidden_dim,
#                          fc_dim=fc_dim,
#                          vocab_size=vocab_size,
#                          tagset_size=target_size,
#                          word_vec_matrix=word_vec_matrix,
#                          dropout=dropout_rate,
#                          max_paragraph_len=max(train_dataset.max_paragraph_length,
#                                                dev_dataset.max_paragraph_length,
#                                                test_dataset.max_paragraph_length))

if GPU:
    model.cuda()

print(model)

weight = torch.Tensor(target_size).float().fill_(0.)

if GPU:
    weight = weight.cuda()

for i in range(mode):
    weight[i] = 100. / train_dataset.emotion_num[i]


optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss(weight=weight, reduce=True, size_average=True)

def train(loader, optimizer, loss_func):
    model.train()

    train_acc = {i: 0. for i in range(target_size)}
    total_loss = 0.
    for batch_times, (word_seq, seq_len, label) in enumerate(loader):
        if batch_times % 100 == 0:
            print("Sentences: ", batch_times * batch_size)

        if GPU:
            word_seq = word_seq.cuda()
            seq_len = seq_len.cuda()
            label = label.cuda()

        targets = label

        sentence_encoder_out, tag_scores = model.forward((word_seq, seq_len))

        pred = torch.max(tag_scores, 1)[1]

        for i in range(len(pred)):
            if pred[i] == targets[i]:
                train_acc[int(targets[i])] += 1

        # loss = loss_func(tag_scores, targets)
        loss, emotion_loss, qa_loss = model.get_loss(sentence_encoder_out=sentence_encoder_out,
                                                     tag_space=tag_scores,
                                                     emotion_loss_func=loss_func,
                                                     targets=targets)

        if batch_times % 100 == 0:
            print("emotion loss: {:.3f} answer loss: {:.3f}".format(float(emotion_loss), float(qa_loss)))
            # print(loss)

        total_loss += float(loss)

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

    total_acc = sum(train_acc[i] for i in range(mode))

    return train_acc, total_acc, total_loss

def eval(loader, loss_func, save_flag=False):
    model.eval()

    acc = {i: 0. for i in range(target_size)}
    total_loss = 0.
    for word_seq, seq_len, label in loader:
        if GPU:
            word_seq = word_seq.cuda()
            seq_len = seq_len.cuda()
            label = label.cuda()

        targets = label

        _, tag_scores = model.forward((word_seq, seq_len))

        pred = torch.max(tag_scores, 1)[1]

        for i in range(len(pred)):
            if pred[i] == targets[i]:
                acc[int(targets[i])] += 1
            else:
                if save_flag and targets[i] < 4:
                    save_mistake_sent.write("pred: {}/{}  {}".format(pred[i], targets[i], ori_sentence(word_seq[i], word_id_dict)))

        total_loss += loss_func(tag_scores, targets)

    total_acc = sum(acc[i] for i in range(mode))

    return acc, total_acc, total_loss

def print_info(sign, total_loss, total_acc, acc, dataset):
    dataset_size = sum(dataset.emotion_num[i] for i in range(mode))
    print("{}: Loss: {:.6f}, Acc: {:.6f}".format(sign, total_loss / dataset_size, total_acc / dataset_size))

    eval = [acc[i] / dataset.emotion_num[i] for i in range(mode)]

    if mode == 4:
        print("{}: 0: {:.6f}, 1: {:.6f}, 2: {:.6f}, 3: {:.6f}".format(sign, eval[0], eval[1], eval[2], eval[3]))
    elif mode == 8:
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

    average_acc = sum(eval) / mode
    print("{}: Average Acc: {:.6f}\n".format(sign, average_acc))

    return average_acc

# train & dev
print("**********************************")
print("train_dataset: ", train_dataset.emotion_num)
print("friends_dev_dataset: ", friends_dev_dataset.emotion_num)
print("emotionpush_dev_dataset: ", emotionpush_dev_dataset.emotion_num)

max_dev_average_acc = [0, 0]
max_dev_average_acc_model_state = [model.state_dict(), model.state_dict()]

max_test_average_acc = [0, 0]
max_test_average_acc_model_state = [model.state_dict(), model.state_dict()]

for epoch in range(epoch_num):
    print("==================================")
    print("epoch: {}".format(epoch))

    # train
    model.train()
    # train_acc, total_acc, total_loss = train(loader=train_loader, loss_func=loss_func, optimizer=optimizer)
    train_acc, total_acc, total_loss = train(loader=train_dataset.get_paragraph(), loss_func=loss_func, optimizer=optimizer)
    train_average_acc = print_info(sign="Train", total_loss=total_loss, total_acc=total_acc, acc=train_acc, dataset=train_dataset)

    # dev
    model.eval()

    for dataset_index in range(2):
        print("----------------------------------")
        # dev_acc, total_acc, total_loss = eval(loader=dev_loader[dataset_index], loss_func=loss_func)
        dev_acc, total_acc, total_loss = eval(loader=dev_dataset[dataset_index].get_paragraph(), loss_func=loss_func)
        dev_average_acc = print_info(sign="Dev_{}".format(dataset_index),
                                     total_loss=total_loss, total_acc=total_acc,
                                     acc=dev_acc, dataset=dev_dataset[dataset_index])

        if train_average_acc > 0.9:
            if dev_average_acc > max_dev_average_acc[dataset_index]:
                max_dev_average_acc[dataset_index] = dev_average_acc
                max_dev_average_acc_model_state[dataset_index] = copy.deepcopy(model.state_dict())
                print("### new max dev acc!\n")
            else:
                print("Dev_{}: Now Max Acc: {:.6f}\n".format(dataset_index, max_dev_average_acc[dataset_index]))

        # tmp check test set
        # test_acc, total_acc, total_loss = eval(loader=test_loader[dataset_index], loss_func=loss_func)
        test_acc, total_acc, total_loss = eval(loader=test_dataset[dataset_index].get_paragraph(), loss_func=loss_func)
        test_average_acc = print_info(sign="Test_{}".format(dataset_index),
                                      total_loss=total_loss, total_acc=total_acc,
                                      acc=test_acc, dataset=test_dataset[dataset_index])

        if train_average_acc > 0.9:
            if test_average_acc > max_test_average_acc[dataset_index]:
                max_test_average_acc[dataset_index] = test_average_acc
                max_test_average_acc_model_state[dataset_index] = copy.deepcopy(model.state_dict())
                print("### new max test acc!\n")
            else:
                print("Test_{}: Now Max Acc: {:.6f}\n".format(dataset_index, max_test_average_acc[dataset_index]))

# test_eval
test_average_acc = [0., 0.]
for index in range(2):
    print("**********************************")
    print("test_dataset_{}: ".format(index), test_dataset[index].emotion_num)

    # load max dev state
    model.load_state_dict(max_dev_average_acc_model_state[index])
    model.eval()

    # test_acc, total_acc, total_loss = eval(loader=test_loader[index], loss_func=loss_func)
    test_acc, total_acc, total_loss = eval(loader=test_dataset[index].get_paragraph(), loss_func=loss_func, save_flag=True)

    test_average_acc[index] = print_info(sign="Test_{}".format(index), total_loss=total_loss,
                                         total_acc=total_acc, acc=test_acc, dataset=test_dataset[index])

# save
torch.save(max_dev_average_acc_model_state[0], save_dir+"friends_max_dev_average_acc_model.pkl")
torch.save(max_dev_average_acc_model_state[1], save_dir+"emotionpush_max_dev_average_acc_model.pkl")

torch.save(max_test_average_acc_model_state[0], save_dir+"friends_max_test_average_acc_model.pkl")
torch.save(max_test_average_acc_model_state[1], save_dir+"emotionpush_max_test_average_acc_model.pkl")