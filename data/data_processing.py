import re
import emoji
import json
import numpy as np
from nltk.tokenize import TweetTokenizer

# utterance process

tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)

def sentence_split(sentence):
    return tweet_tokenizer.tokenize(sentence)

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + hashtag_body.split(r"(?=[A-Z])"))
    return result

def emojis(text):
    text = text.encode('utf-16', 'surrogatepass').decode('utf-16')
    for word in text:
        if word in emoji.UNICODE_EMOJI:
            emoji_desc = emoji.demojize(word)
            plain_word = re.sub(r"[^a-z]", r" ", emoji_desc)
            text = re.sub(word, plain_word, text)
    return text

def utterance_process(text):
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    text = text.lower()

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=re.MULTILINE | re.DOTALL)

    text = re_sub(r"/n", " ")
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/", " / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lol>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sad>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutral>")
    text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    # replace person and location
    text = text.replace("person_<number>", "person")
    text = text.replace("location_<number>", "location")

    # find emoji
    text = emojis(text)

    text = re_sub(r"(\w)\1{2,}(\S*)\b", r"\1\2 <repeat>")
    try:
        # remove unicode
        text = re.sub(r"\u0092|\x92", "'", text)
        text = text.encode("utf-8").decode("ascii", "ignore")
    except:
        pass
    # split with punctuations
    text = re_sub(r"([^A-Za-z0-9\_]+)", r" \1 ")
    text = sentence_split(text)

    text = " ".join(text)

    # text = re_sub(r"< (.*?) >", r"<\1>")

    return text

# emotion process

def emotion2label(string):
    s = {
        'neutral': 0,
        'joy': 1,
        'sadness': 2,
        'anger': 3,
        'disgust': 4,
        'fear': 5,
        'surprise': 6,
        'non-neutral': 7
    }

    # s = {
    #     'neutral': 0,
    #     'joy': 1,
    #     'sadness': 2,
    #     'anger': 3,
    #     'disgust': 3,
    #     'fear': 3,
    #     'surprise': 1,
    #     'non-neutral': 0
    # }

    num = s.get(string)
    return num
    # a = np.zeros([8])
    # a[num] = 1
    # return a

# speaker process

speaker_names = {}

def speaker2id(name):
    if name not in speaker_names:
        speaker_names[name] = len(speaker_names)
    return speaker_names[name]

# utterance to sequence

word_dictionary = {}

def utterance2sequence(sentence):
    ret = ""
    word_list = sentence.split()
    if len(word_list) == 0:
        word_list.append(".")
    for word in word_list:
        ret = ret + str(word_dictionary[word]) + " "
    return ret

# save

def get_precessed_data(data_dir):
    data_file = open(data_dir, "r", encoding="utf-8")
    data_json = json.load(data_file)

    for dialog in data_json:
        for sentence in dialog:
            sentence["speaker"] = speaker2id(sentence["speaker"])
            sentence["utterance"] = utterance_process(sentence["utterance"])
            sentence["emotion"] = emotion2label(sentence["emotion"])

            for word in sentence["utterance"].split():
                if word not in word_dictionary:
                    word_dictionary[word] = len(word_dictionary) + 1

    return data_json

def get_data_sequence(proc_json):
    for dialog in proc_json:
        for sentence in dialog:
            sentence["utterance"] = utterance2sequence(sentence["utterance"])
    return proc_json

def build_word_vec(save_dir, word_vec_dir):
    print("Start Build Word Vec ...")
    save_file = open(save_dir, "w", encoding="utf-8")
    save_file.write(str(len(word_dictionary))+"\n")

    word_vec_file = open(word_vec_dir, "r", encoding="utf-8")

    found_num = 0

    for num, line in enumerate(word_vec_file):
        if num % 100000 == 0:
            print("line {} ...".format(num))
        word, vec = line.split(' ', 1)
        if word in word_dictionary:
            save_file.write("{} {}\n".format(word_dictionary[word], vec))
            found_num += 1

    print("Found {}/{} words with glove vectors".format(found_num, len(word_dictionary)))


if __name__ == '__main__':
    GloVe_dir = "D:/Documents/Python_Project/WordVector_Folder/glove.840B.300d.txt"
    DataSet = "Merge"
    DataSet_Tag = ["train", "test", "dev"]

    DataSet_json = {}

    if DataSet != "Merge":
        for tag in DataSet_Tag:
            DataSet_json[tag] = get_precessed_data("./{}/{}_{}.json".format(DataSet, DataSet.lower(), tag))
    else:
        for tag in DataSet_Tag:
            DataSet_json[tag] = get_precessed_data("./Friends/friends_{}.json".format(tag)) + \
                                get_precessed_data("./EmotionPush/emotionpush_{}.json".format(tag))

    for tag in DataSet_Tag:
        data_seq_json = get_data_sequence(DataSet_json[tag])
        json.dump(data_seq_json, open("./{}_Proc/{}_seq_{}.json".format(DataSet, DataSet.lower(), tag),
                                      "w", encoding="utf-8"))

    build_word_vec("./{}_Proc/{}_word_vec.txt".format(DataSet, DataSet.lower()), GloVe_dir)
