#coding=utf-8
import os
import tensorflow as tf
from tqdm import tqdm
from model.gpt2 import TFGPT2Model
from model.gpt2_config import GPT2Config
from model.data_helper import gen_voc,load_vocab,load_traindata_ids,gen_batch_data
from model.model_helper import CustomSchedule,loss_function

raw_path = './data'
save_vocab_path = './vocab/vocab.txt'
checkpoint_path='./checkpoint/train/'

def creat_model(config):
    gpt2model = TFGPT2Model(config)
    if config.dynamics_lr:
        learning_rate = CustomSchedule(config.n_embd)
    else:
        learning_rate = config.lr
    optimizer = tf.keras.optimizers.Adam(
                                    learning_rate,
                                    beta_1=0.9,
                                    beta_2=0.98,
                                    epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    '''
    训练过程中查看信息
    '''
    return gpt2model,optimizer,loss_object

if __name__ == '__main__':
    '''
    字典未生成则生成字典，生成了字典则载入字典
    '''
    if not os.path.exists(save_vocab_path):
        gen_voc(raw_path,save_vocab_path)
    word2id,id2word = load_vocab(save_vocab_path)
    '''
    载入相关配置信息
    '''
    config = GPT2Config()    
    '''
    训练代码
    '''
    gpt2model,optimizer,loss_object = creat_model(config)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    '''
    载入旧模型
    '''
    if tf.train.latest_checkpoint(checkpoint_path) is not None:
        print("recover old model....")
        gpt2model.load_weights(tf.train.latest_checkpoint(checkpoint_path))
    else:
        print("creat new model....")
    # 指定input_signature何时调用tf.function以确保仅构建一个功能图
    train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
    @tf.function(input_signature=train_step_signature)
    def train_step(input_ids):
        with tf.GradientTape() as tape:
            outputs = gpt2model(input_ids[:, :-1],training=True)
            logits = outputs[0]
            target = input_ids[:, 1:]
            loss = loss_function(target, logits, loss_object)
        gradients = tape.gradient(loss, gpt2model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, gpt2model.trainable_variables))

        train_loss(loss)
        train_accuracy(target,logits)
    '''
    运行训练过程
    '''
    all_step = 0
    epoch = 0
    data_loop = 0
    finished_files = []
    while epoch < config.epoch:
        train_loss.reset_states()
        train_accuracy.reset_states()
        '''
        训练的id数据载入
        '''
        ids,reach_end,finished_files = load_traindata_ids(raw_path,
                                           word2id,
                                           config.n_ctx,
                                           config.read_len,
                                           data_loop,
                                           finished_files)
        data_loop = data_loop + 1
        '''
        开始训练模型
        '''
        for index,batch in enumerate(tqdm(gen_batch_data(config.batch_size,ids))):
            train_step(batch)
            if index % 50 == 0 and index > 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, index, train_loss.result(), train_accuracy.result()))
            if index % 10000 == 0 and index > 0:
                gpt2model.save_weights(checkpoint_path + "gpt2-" + str(all_step))
                print('Saving checkpoint inner for epoch {}'.format(epoch+1))
            all_step = all_step + 1
        gpt2model.save_weights(checkpoint_path + "gpt2-" + str(all_step))
        print('Saving checkpoint outter for epoch {}'.format(epoch+1))
        if reach_end or len(ids) == 0:
            '''
            下一个epoch数据重新开始
            '''
            print(finished_files)
            epoch = epoch + 1
            data_loop = 0
            finished_files = []