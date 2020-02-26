#coding=utf-8
import os
import numpy as np
import tensorflow as tf
from model.gpt2 import TFGPT2Model
from model.gpt2_config import GPT2Config
from model.data_helper import load_vocab
from model.infer_helper import convert2ids,ids2text,gen_sequence

save_vocab_path = './vocab/vocab.txt'
checkpoint_path='./checkpoint/train'

def interact_model(
    dialog_history=5,
    batch_size=1,
    temperature=1,
    top_k=0,
    top_p=1):
    '''
    载入字典
    '''
    word2id,id2word = load_vocab(save_vocab_path)
    '''
    载入模型和参数
    '''
    config = GPT2Config()
    gpt2_model = TFGPT2Model(config)
    checkpoint = tf.train.Checkpoint(gpt2_model=gpt2_model)
    ckpt = tf.train.latest_checkpoint(checkpoint_path)
    if ckpt == None:
        print("no ckpt exist!!!")
        raise
    checkpoint.restore(ckpt)
    '''
    执行对话生成
    '''
    @tf.function
    def infer_step(context):
        output = gen_sequence(model=gpt2_model,
                            length=config.n_ctx,
                            context=context,
                            eos_token=word2id['SEP'],
                            batch_size=batch_size,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p)
        return output
    history_text = []
    while True:
        raw_text = input("user input>>")
        while not raw_text:
            print('输入为空，重新输入')
            raw_text = input("user input:")
        context_tokens = convert2ids(raw_text,word2id,config.n_ctx)
        history_text.append(context_tokens)
        infer_data = []
        infer_data.append(word2ids['SOS'])
        for item in history_text:
            infer_data.extend(item)
        infer_data = np.array(infer_data,dtype=np.int64)
        '''
        修改维度为[1,len(raw_ids)]以适应transformer的运算维度
        '''
        infer_data = np.expand_dims(infer_data,0)
        out = infer_step(infer_data)
        history_text.append(out[-1])
        text = ids2text(out[-1])
        print("robot>>{}\n".format(text))
        print("*" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)