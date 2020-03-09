#coding=utf-8
import tensorflow as tf
import numpy as np

def convert2ids(raw_inputs,word2ids):
    ids = []
    for item in raw_inputs:
        id_data = word2ids["UNK"]
        if item in word2ids.keys():
            id_data = word2ids[item]
        ids.append(id_data)
    return ids

def ids2text(output,id2word):
    SEP = 5
    if output[-1] == SEP:
        output = output[:-1]
    return ''.join([id2word[item] for item in output])

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    if min_values.shape.as_list() == [0]:
        min_values = tf.constant([0.0])
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,)

def gen_sequence(model=None,
                length=None,
                context=None,
                past_shape=None,
                start_token=None,
                eos_token=None,
                batch_size=None,
                vocab_size=None,
                temperature=1,
                top_k=0,
                top_p=1):
    '''
    若要进行无条件随机输入，start_token赋值SOS，输入为SOS_id。否则未context
    '''
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)
    '''
    生成logits和past
    '''
    def step(tokens, past=None):
        lm_output = model(inputs=tokens, past=past, training=False)

        logits = lm_output[0][:, :, :vocab_size]
        if past_shape is None:
            return {'logits': logits}
        else:
            presents = lm_output[1]
            presents.set_shape(past_shape)
            return {'logits': logits,'presents': presents}
    '''
    可以看到每次输入的prev是前一个samples，只有一个字，所以每次取logit最后一个输出
    而prev则每次记录前面的输出
    '''
    with tf.name_scope('sample_sequence'):
        init_len = context.shape[-1]
        output = tf.Variable([[]],shape=(1,None),dtype=tf.int32)
        prev = tf.cast(context,dtype=tf.int32)
        
        def body(prev, output,past=None):
            next_outputs = step(prev, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.cast(temperature,dtype=tf.float32)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            output = tf.concat([output, samples], axis=1)
            if past_shape is None:
                output = tf.cast(output,dtype=tf.int32)
                prev = tf.cast(prev,dtype=tf.int32)
                next_input = tf.concat([prev,samples],axis=1) 
            else:
                next_input = samples

            return [next_input,output]

        def cond(*arg):
            output = arg[-1]
            stop_token = tf.constant(eos_token,dtype=tf.int32)
            if output.shape[-1] == None:
                return True
            else:
                if output[-1][-1] == stop_token:
                    return False
                else:
                    return True

        prev, output = tf.while_loop(
                        cond,
                        body,
                        [prev, output],
                        maximum_iterations=tf.constant(length - init_len - 1, dtype=tf.int32)
                        )

        return output