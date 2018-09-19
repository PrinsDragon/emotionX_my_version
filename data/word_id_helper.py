def read_word_id(word_id_dir):
    word_id_dict = {}
    file = open(word_id_dir, "r", encoding="utf-8")
    for line in file:
        id, word = line.split(" ")
        id = int(id)
        word_id_dict[id] = word.replace("\n", " ")
    return word_id_dict

def ori_sentence(sentence, word_id_dict):
    ret = ""
    for id in sentence:
        id = int(id)
        if id == 0:
            break
        word = word_id_dict[id]
        ret = ret + word
    ret = ret+"\n"
    return ret
