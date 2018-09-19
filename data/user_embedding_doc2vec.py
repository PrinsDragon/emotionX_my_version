import os

import pandas as pd
import numpy as np
import csv
import json
import gensim

# region proc_func

def generate_user_paragragh(input_file_path, output_file_path):
    input_file = open(input_file_path, "r")
    json_data = json.load(input_file)

    data_dict = {}
    for dialog in json_data:
        for sentence in dialog:
            speaker = sentence["speaker"]
            utterance = sentence["utterance"]

            if speaker in data_dict:
                data_dict[speaker] = data_dict[speaker] + utterance + " <END> "
            else:
                data_dict[speaker] = utterance + " <END> "

    output_file = open(output_file_path, "w")
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

def save_model(data, model, output_path):
    output_file = open(output_path, "w")
    for line in data:
        speaker = line[0]
        sentence = line[1]
        vector = model.infer_vector(sentence)
        vector = list(vector)
        output_file.write("{} {}\n".format(speaker, vector))


def doc2vec(input_csv_path, kind):

    data = np.asarray(pd.read_csv(input_csv_path, header=None))
    speaker_num = len(data)
    doc_labels = [data[i][0] for i in range(speaker_num)]
    docs = [data[i][1] for i in range(speaker_num)]
    it = LabeledLineSentence(docs, doc_labels)

    model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025)
    model.build_vocab(it)
    for epoch in range(50):
        print("{} Epoch: {}".format(kind, epoch))
        model.alpha -= 0.002
        model.min_alpha = model.alpha
        model.train(it, total_examples=model.corpus_count, epochs=model.iter)

    save_model(data, model, "./Merge_Proc/user_embedding/merge_user_embedding_{}.txt".format(kind))

    print("==============================================")

# endregion

if __name__ == '__main__':
    input_kind = ["emotionpush_dev",
                  "emotionpush_test",
                  "friends_dev",
                  "friends_test",
                  "train"]
    for kind in input_kind:
        input_file_path = "./Merge_Proc/merge_seq_{}.json".format(kind)
        output_file_path = "./Merge_Proc/user_paragraph/merge_user_para_{}.csv".format(kind)

        generate_user_paragragh(input_file_path, output_file_path)

        doc2vec(output_file_path, kind)
