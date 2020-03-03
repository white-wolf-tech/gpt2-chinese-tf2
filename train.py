#coding=utf-8
import os
import tensorflow as tf
from tqdm import tqdm
from model.gpt2 import TFGPT2Model
from model.gpt2_config import GPT2Config
from model.data_helper import gen_voc,load_vocab,process_raws_data,load_traindata_ids,gen_batch_data
from model.model_helper import CustomSchedule,checkmodel,loss_function

raw_path = './data'
save_vocab_path = './vocab/vocab.txt'
checkpoint_path='./checkpoint/train'
checkpoint_prefix = os.path.join(checkpoint_path, "gpt2")

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
    ckpt_manager = checkmodel(checkpoint_path,gpt2model,optimizer)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    '''
    训练过程中查看信息
    '''
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    return gpt2model,optimizer,ckpt_manager,loss_object,train_loss,train_accuracy

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
    训练的id数据未生成则生成id数据
    '''
    if not os.path.exists(raw_path + '/ids'):
        process_raws_data(raw_path,word2id,config.n_ctx)
    elif len(os.listdir(raw_path + '/ids')) == 0:
        process_raws_data(raw_path,word2id,config.n_ctx)
    ids = load_traindata_ids(raw_path)
    '''
    加载模型
    '''
    gpt2model,optimizer,ckpt_manager,loss_object,train_loss,train_accuracy = creat_model(config)
    '''
    训练代码
    '''
    train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
    @tf.function(input_signature=train_step_signature)
    def train_step(input_ids):
        with tf.GradientTape() as tape:
            outputs = gpt2model(input_ids,training=True)
            logits = outputs[0][:, :-1,:]
            target = input_ids[:, 1:]
            loss = loss_function(target, logits, loss_object)
        gradients = tape.gradient(loss, gpt2model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, gpt2model.trainable_variables))

        train_loss(loss)
        train_accuracy(target,logits)
    '''
    运行训练过程
    '''
    for epoch in range(config.epoch):
        train_loss.reset_states()
        train_accuracy.reset_states()
        for index,batch in enumerate(tqdm(gen_batch_data(config.batch_size,ids))):
            train_step(batch)
            if index % 50 == 0 and index > 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, index, train_loss.result(), train_accuracy.result()))
            if index % 500 == 0 and index > 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint inner for epoch {} at {}'.format(epoch+1,ckpt_save_path))
                print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(),train_accuracy.result()))
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint outter for epoch {} at {}'.format(epoch+1,ckpt_save_path))
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(),train_accuracy.result()))