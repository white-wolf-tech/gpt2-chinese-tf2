import tensorflow as tf
import numpy as np
from model.infer_helper import convert2ids,ids2text


save_vocab_path = './vocab/vocab.txt'
def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    '''
    打印计算图中的名称
    '''
    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def load_run_pb(pb_path="./checkpoint/chat.pb"):
    # 使用tf2.0中集成的tf1.0函数载入pb文件
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # 从pb中恢复tf.function的concrete函数
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["input:0"],
                                    outputs=["output:0"],
                                    print_graph=True)

    word2id,id2word = load_vocab(save_vocab_path)
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
        out = frozen_func(x=infer_data)
        '''
        当前robot输出结果存入对话buffer
        '''
        history_buffer.append(out,id2word)
        '''
        解码并且显示结果
        '''
        text = ids2text(out[-1])
        print("robot>>{}\n".format(text))
        print("*" * 80)
if __name__ == '__main__':
    load_run_pb()