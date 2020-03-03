#coding=utf-8
import os
import random
from tqdm import tqdm
import numpy as np

'''
检查目录文件
'''
def get_files(path,code):
    files = []
    if os.path.isdir(path):
        all_files = os.listdir(path)
        if len(all_files) == 0:
            print('train data is empty')
            raise
        for file in all_files:
            if not file.endswith(code):
                continue
            if path.endswith('/'):
                files.append(path + file)
            else:
                files.append(path + '/' + file)
    else:
        print("the data dir is not correct")
        raise
    return files
'''
字典生成以及字典的载入
'''
def gen_voc(raw_path,save_vocab_path):
    vocab = ['PAD','unused0','unused1','UNK','SOS','SEP']
    for data_path in get_files(raw_path,'.txt'):
        with open(data_path,'rb') as f:
            str_temp = f.read().decode('utf-8')
            if '\r\n' in str_temp:
                str_temp = str_temp.replace('\r\n','')
            elif '\n' in str_temp:
                str_temp = str_temp.replace('\n','')
            for data in str_temp:
                if data not in vocab:
                    vocab.append(data)
                else:
                    pass
    with open(save_vocab_path,'w') as wf:
        wf.write('\n'.join(vocab))

def load_vocab(save_vocab_path):
    with open(save_vocab_path,'r') as f:
        vocab = f.read().strip('\n').strip().split('\n')
        word2id = dict(zip(vocab,list(range(len(vocab)))))
        id2word = dict(zip(list(range(len(vocab))),vocab))
        return word2id,id2word
'''
生成训练数据相关代码
'''
def preprocess_triandata2ids(train_raw_path, train_tokenized_path, word2id, n_ctx):
    max_len = 0
    count_beyond = 0
    with open(train_raw_path, 'rb') as f:
        data = f.read().decode("utf-8")
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    with open(train_tokenized_path, "w", encoding="utf-8") as f:
        for dialogue_index, dialogue in enumerate(tqdm(train_data)):
            if "\r\n" in data:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            dialogue_ids = [str(word2id['SOS'])]  # 每句话以SOS开头
            for utterance in utterances:
                for word in utterance:
                    if word in word2id.keys():
                        dialogue_ids.extend([str(word2id[word])])
                    else:
                        dialogue_ids.extend([str(word2id['unused0'])])
                dialogue_ids.append(str(word2id['SEP']))  #每句话结束加SEP
            # 超过最大长度的数据则丢弃
            if len(dialogue_ids) > max_len:
                max_len = len(dialogue_ids)
            if len(dialogue_ids) > n_ctx:
                count_beyond = count_beyond + 1
                continue
            #小于最大长度的数据写入ids文件
            f.write(' '.join(dialogue_ids))
            # 最后一条记录不添加换行符
            if dialogue_index < len(train_data) - 1:
                f.write("\n")
    print("tokenized_file is stored in {}".format(train_tokenized_path))
    print("bigger than max len count is {}".format(count_beyond))
    return max_len

def process_raws_data(path,word2id, n_ctx):
    max_len_end = 0
    for file in get_files(path,'.txt'):
        temp_path = file.split('/')
        tokenized_name = temp_path[-1].split('.txt')[0] + '.ids'
        tokenized_path = '/'.join(temp_path[:-1]) + '/ids/'
        if not os.path.exists(tokenized_path):
            os.mkdir(tokenized_path)
        tokenized_file = tokenized_path + tokenized_name
        max_len = preprocess_triandata2ids(file,tokenized_file,word2id,n_ctx)
        if max_len > max_len_end:
            max_len_end =  max_len
    print("max len in train corpus is {}".format(max_len_end))

'''
载入训练数据相关代码
'''
def load_traindata_ids(path):
    if path.endswith('/'):
        tokenized_path = path + 'ids/'
    else:
        tokenized_path = path + '/ids/'
    if not os.path.exists(tokenized_path):
        print("train data ids dir is not exist")
        raise
    tokenized_files = get_files(tokenized_path,'.ids')
    ids = []
    for item in tokenized_files:
        with open(item,'r') as f:
            datas = f.read().strip().strip('\n').split('\n')
            for token in datas:
                token = token.split(' ')
                token = [int(item) for item in token]
                ids.append(token)
    return ids

def padding2maxlen(batch):
    PAD = 0
    max_item = sorted(batch, key=lambda x: len(x),reverse=True)[0]
    max_len = len(max_item)
    for index,item in enumerate(batch):
        if len(item) < max_len:
            batch[index] = item + [PAD] * (max_len - len(item))
    return batch

def gen_batch_data(batch_size,ids):
    random.shuffle(ids)
    ids_len = len(ids)
    batchs = []
    index = 0
    while index < ids_len:
        if index + batch_size < ids_len:
            batch_item = padding2maxlen(ids[index : index + batch_size])
        else:
            batch_item = padding2maxlen(ids[index : ])
        batchs.append(np.array(batch_item,dtype=np.int64))
        index = index + batch_size
    return batchs
