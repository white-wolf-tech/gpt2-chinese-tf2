#coding=utf-8
import tensorflow as tf
import numpy as np

def convert2ids(raw_inputs,word2ids):
    return [word2ids[item] for item in raw_inputs]

def ids2text(id2word,output):
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
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )

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
        presents = lm_output[1]
        presents.set_shape(past_shape)
        return {'logits': logits,'presents': presents}
    '''
    可以看到每次输入的prev是前一个samples，只有一个字，所以每次取logit最后一个输出
    而prev则每次记录前面的输出
    '''
    with tf.name_scope('sample_sequence'):
        def body(past, prev, output):
            next_outputs = step(prev, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1)
            ]
        '''
        初始化body函数
        '''
        past, prev, output = body(None, context, context)

        def cond(*args):
            if eos_token!=None:
                return (output[-1] == eos_token)
            else:
                return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                tf.TensorShape(past_shape),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens
