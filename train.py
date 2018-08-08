#coding=utf-8

import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Net import BiLSTM_BiLSTM

GPU = torch.cuda.is_available()
print("GPU available: ", GPU)

# parameters
# dataset = "EmotionPush"
dataset = "Merge"
mode = 4
print("Now running dataset = {}, mode = {}".format(dataset, mode))

train_dir = "./data/{}_Proc/{}_seq_train.json".format(dataset, dataset.lower())
dev_dir = "./data/{}_Proc/{}_seq_dev.json".format(dataset, dataset.lower())
test_dir = "./data/{}_Proc/{}_seq_test.json".format(dataset, dataset.lower())
word_vector_dir = "./data/{}_Proc/{}_word_vec.txt".format(dataset, dataset.lower())

epoch_num = 50
embedding_dim = 300
hidden_dim = 300
fc_dim = 128
batch_size = 128
gradient_max_norm = 5
target_size = 8
dropout_rate = 0.8

print("epoch_num: ", epoch_num)

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
            #np.zeros((word_num, embedding_dim))
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

    def extend(self, total_length):
        tmp = torch.zeros(total_length-self.seq_len).long()
        self.seq = torch.cat([self.seq, tmp], 0)

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

def train(model, loader, optimizer, loss_func):
    train_acc = {i: 0. for i in range(target_size)}
    total_acc = 0.
    total_loss = 0.
    for batch_times, (batch_x, batch_x_len, batch_y) in enumerate(loader):
        if batch_times % 20 == 0:
            print("Sentences: ", batch_times * batch_size)

        if GPU:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_x_len = batch_x_len.cuda()

        sentence_in = (batch_x, batch_x_len)
        targets = batch_y

        tag_scores = model(sentence_in)

        pred = torch.max(tag_scores, 1)[1]

        total_acc += float((pred == targets).sum())

        for i in range(len(pred)):
            if pred[i] == targets[i]:
                train_acc[int(targets[i])] += 1

        # loss = loss_func(tag_scores, targets)

        loss = model.get_loss(sentence_tuple=sentence_in,
                              emotion_loss_func=loss_func,
                              targets=targets)

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

    return train_acc, total_acc, total_loss

def eval(model, loader, loss_func):
    model.eval()

    acc = {i: 0. for i in range(target_size)}
    total_acc = 0.
    total_loss = 0.
    for batch_x, batch_x_len, batch_y in loader:

        if GPU:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_x_len = batch_x_len.cuda()

        sentence_in = (batch_x, batch_x_len)
        targets = batch_y

        tag_scores = model(sentence_in)

        pred = torch.max(tag_scores, 1)[1]

        total_acc += float((pred == targets).sum())

        for i in range(len(pred)):
            if pred[i] == targets[i]:
                acc[int(targets[i])] += 1

        total_loss += loss_func(tag_scores, targets)

    return acc, total_acc, total_loss

def print_info(sign, total_loss, total_acc, acc, dataset):
    print("{}: Loss: {:.6f}, Acc: {:.6f}".format(sign, total_loss / (len(dataset)), total_acc / (len(dataset))))

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

vocab_size, word_vec_matrix = build_word_vec_matrix(word_vector_dir)

model = BiLSTM_BiLSTM(embedding_dim=embedding_dim,
                      hidden_dim=hidden_dim,
                      fc_dim=fc_dim,
                      vocab_size=vocab_size,
                      tagset_size=target_size,
                      word_vec_matrix=word_vec_matrix,
                      dropout=dropout_rate)

if GPU:
    model.cuda()

print(model)

weight = torch.Tensor(target_size).float().fill_(0.)

if GPU:
    weight = weight.cuda()

for i in range(mode):
    weight[i] = 100. / train_dataset.emotion_num[i]


optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss(weight=weight, reduce=True, size_average=False)

# train & dev
print("**********************************")
print("train_dataset: ", train_dataset.emotion_num)
print("dev_dataset: ", dev_dataset.emotion_num)

max_dev_average_acc = 0
max_dev_average_acc_model_state = model.state_dict()

max_test_average_acc = 0
max_test_average_acc_model_state = model.state_dict()

for epoch in range(epoch_num):
    print("----------------------------------")
    print("epoch: {}".format(epoch))

    # train
    model.train()
    train_acc, total_acc, total_loss = train(model=model, loader=train_loader, loss_func=loss_func, optimizer=optimizer)
    print_info(sign="Train", total_loss=total_loss, total_acc=total_acc, acc=train_acc, dataset=train_dataset)

    # dev
    model.eval()
    dev_acc, total_acc, total_loss = eval(model=model, loader=dev_loader, loss_func=loss_func)
    dev_average_acc = print_info(sign="Dev", total_loss=total_loss, total_acc=total_acc, acc=dev_acc, dataset=dev_dataset)

    if dev_average_acc > max_dev_average_acc:
        max_dev_average_acc = dev_average_acc
        max_dev_average_acc_model_state = model.state_dict()
        print("### new max dev acc!\n")
    else:
        print("Dev: Now Max Acc: {:.6f}\n".format(max_dev_average_acc))

    # tmp check test set
    test_acc, total_acc, total_loss = eval(model=model, loader=test_loader, loss_func=loss_func)
    test_average_acc = print_info(sign="Test", total_loss=total_loss, total_acc=total_acc, acc=test_acc, dataset=test_dataset)

    if test_average_acc > max_test_average_acc:
        max_test_average_acc = test_average_acc
        max_test_average_acc_model_state = model.state_dict()
        print("### new max test acc!\n")
    else:
        print("Test: Now Max Acc: {:.6f}\n".format(max_test_average_acc))

print("epoch = {} max dev acc = {:.6f}\n".format(epoch_num, max_dev_average_acc))
print("epoch = {} max test acc = {:.6f}\n".format(epoch_num, max_test_average_acc))

# test_eval
print("**********************************")
print("test_dataset: ", test_dataset.emotion_num)

# load max dev state
model.load_state_dict(max_dev_average_acc_model_state)

test_acc, total_acc, total_loss = eval(model=model, loader=test_loader, loss_func=loss_func)

print_info(sign="Test", total_loss=total_loss, total_acc=total_acc, acc=test_acc, dataset=test_dataset)

