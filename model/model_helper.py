#coding=utf-8
import tensorflow as tf
'''
学习率生成模块
'''
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=16000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

'''
检查是否能载入训练好的模型
'''
def checkmodel(checkpoint_path,model,opt):
    ckpt = tf.train.Checkpoint(model=model,optimizer=opt)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    else:
        print ('create new model...')
    return ckpt_manager
'''
计算loss
'''
def loss_function(target, logits,loss_object):
    padding = 0
    mask = tf.math.logical_not(tf.math.equal(target, padding))
    loss_ = loss_object(target, logits)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
