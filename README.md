# tensorflow2.0 for Chinese GPT2 

## UPDATE 2020.02.25
参考使用 HuggingFace的[transformers](https://github.com/huggingface/transformers)实现GPT2模型的编写与训练。
使用50W中文闲聊语料进行训练。目前只写了训练代码，后面添加infer代码。

## 运行环境
python3.6、 tensorflow==2.1.0


##训练说明
训练时，将一条训拼接，如 **"[SOS]四级过了没？[SEP]两次都只差多分。[SEP]心疼你三秒钟[SEP]不着急，慢慢来。急不来的，[SEP]
你慢慢吧我着急六级[SEP]人家四级没过你就要过六级了。[SEP]都加油[SEP]加油！[SEP]"**
```
四级过了没？
两次都只差多分。
心疼你三秒钟
不着急，慢慢来。急不来的，
你慢慢吧我着急六级
人家四级没过你就要过六级了。
都加油
加油！
```
50W条语料在data/目录下，data/ids目录是50W条语料转id后的结果
