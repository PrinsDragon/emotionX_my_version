import os

import pandas as pd
import numpy as np
import csv
import json
import gensim

speaker_id_dict = {}
speaker_id_file_path = ""
speaker_id_file = open(speaker_id_file_path, "r")
for line in speaker_id_file:
    name_id, name = line.split(' ', 1)
    speaker_id_dict[name] = name_id

def generate_user_paragragh(input_file_path, output_file_path):
    input_file = open(input_file_path, "r")
    json_data = json.load(input_file)

    data_dict = {}
    for dialog in json_data:
        for sentence in dialog:
            speaker = sentence["speaker"]
            speaker_id = speaker_id_dict[speaker]
            utterance = sentence["utterance"]

            if speaker_id in data_dict:
                data_dict[speaker_id] = data_dict[speaker_id] + utterance + " <END> "
            else:
                data_dict[speaker_id] = utterance + " <END> "

    output_file = open(output_file_path)
    writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)

    for speaker_id in data_dict:
        line = [speaker_id, data_dict[speaker_id]]
        writer.writerow(line)


TaggedDocument = gensim.models.doc2vec.TaggedDocument

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        count = 0
        for idx, doc in enumerate(self.doc_list):
            try:
                yield TaggedDocument(doc.split(), [self.labels_list[idx]])
            except AttributeError:
                count += 1

def doc2vec(input_csv_path):

    data = np.asarray(pd.read_csv(input_csv_path, header=None))
    speaker_num = len(data)
    doc_labels = [data[i][0] for i in range(speaker_num)]
    docs = [data[i][1] for i in range(speaker_num)]
    it = LabeledLineSentence(docs, doc_labels)

    model = gensim.models.Doc2Vec(size=100, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025)
    model.build_vocab(it)
    for epoch in range(50):
        print("Epoch: " + str(epoch))
        model.alpha -= 0.002
        model.min_alpha = model.alpha
        model.train(it, total_examples=model.corpus_count, epochs=model.iter)
        directory = "./models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        model.save(directory + "/user_stylometric.model")
