#coding=utf-8
import os
import fire
import numpy as np
import tensorflow as tf
from model.gpt2 import TFGPT2Model
from model.gpt2_config import GPT2Config
from model.data_helper import load_vocab
from model.infer_helper import convert2ids,ids2text,gen_sequence
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

save_vocab_path = './vocab/vocab.txt'
checkpoint_path='./checkpoint/train'

def interact_model(
    dialog_history=10,
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
    '''
    获取计算图
    '''
    gpt2_concrete = infer_step.get_concrete_function(
        context = tf.TensorSpec(shape=(None, None), dtype=tf.int64))
    '''
    获取静态图计算函数
    '''
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    '''
    打印计算节点
    '''
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("*" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("*" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    '''
    打包pb，并写入文件
    '''
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./checkpoint",
                      name="chat.pb",
                      as_text=False)
    '''
    chat bot 开始
    '''
    history_buffer = []
    while True:
        '''
        超过buffer容量
        '''
        if len(history_buffer) > dialog_history:
            history_buffer = []

        raw_text = input("user input>>")
        while not raw_text:
            print('输入为空，重新输入')
            raw_text = input("user input:")
        '''
        获取输入ids
        '''
        context_tokens = convert2ids(raw_text,word2id)
        history_buffer.append(context_tokens)
        infer_data = []
        infer_data.append(word2ids['SOS'])
        for item in history_buffer:
            infer_data.extend(item)
            infer_data.append(word2ids['SEP'])
        '''
        检查输入是否超过最大长度，超过则清空buffer，重新输入数据
        '''
        if len(infer_data) > config.n_ctx:
            continue
        '''
        修改维度为[1,len(infer_data)]以适应transformer的运算维度
        '''
        infer_data = np.array(infer_data,dtype=np.int64)
        infer_data = np.expand_dims(infer_data,0)
        '''
        执行inference
        '''
        out = infer_step(infer_data)
        '''
        当前robot输出结果存入对话buffer
        '''
        history_buffer.append(out[-1])
        '''
        解码并且显示结果
        '''
        text = ids2text(out[-1],id2word)
        print("robot>>{}\n".format(text))
        print("*" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)