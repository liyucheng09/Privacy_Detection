import tensorflow as tf
import bert4keras
import keras
import os
print(tf.__version__)
print(keras.__version__)
print(bert4keras.__version__)
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
from  utils import utils
from imp import reload
reload(utils)

train_data = utils.load_data('../data/train.txt')
valid_data = utils.load_data('../data/val.txt')

ret_train_lst = []
for data in train_data:
    ret = 0
    for char,_ in data:
        ret += len(char)
    ret_train_lst.append(ret)
ret_val_lst = []
for data in valid_data:
    ret = 0
    for char,_ in data:
        ret += len(char)
    ret_val_lst.append(ret)

print(max(ret_train_lst),min(ret_train_lst))
print(max(ret_val_lst),min(ret_val_lst))
print(len(ret_train_lst))
print(len(ret_val_lst))

# 预训练模型的超参数
maxlen = 300
epochs = 15
batch_size = 8
bert_layers = 12
learing_rate = 1e-5 
crf_lr_multiplier = 1000 
rnn_lr_multiplier = 1000
# bert配置
config_path = '../model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../model/chinese_L-12_H-768_A-12/vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射
labels = [
    'position',
    'name',
    'organization',
    'company',
    'address',
    'movie',
    'game',
    'government',
    'scene',
    'book',
    'mobile',
    'email',
    'QQ',
    'vx',
]

# 0 表示 'O'
# 其他数字表示对应的 B 和 I
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1

model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)

CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)
model = Model(model.input, output)
print(model.summary())

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learing_rate),
    metrics=[CRF.sparse_accuracy]
)

class data_generator(DataGenerator):
    """
    数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        # is_end  是否到 epoch 的末尾， item 对应的 train_data[i]
        for is_end, item in self.sample(random):
            # vocab.txt 从 0 开始计数，0:[pad] 1-99:[unused*] 100:[UNK] [CLS]:101
            # token_ids:101
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                # 得到除了 [CLS] [SEP] 之外的每一个词的 ids
                # e.g '全国青联委员经纪人' -> [1059, 1744, 7471, 5468, 1999, 1447, 5307, 5279, 782]
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    # e.g [101, 1059, 1744, 7471, 5468, 1999, 1447, 5307, 5279, 782]
                    # 把 [CLS] 加在前面
                    token_ids += w_token_ids
                    if l == 'O':
                        # 如果是 'O' 则标签是 0
                        labels += [0] * len(w_token_ids)
                    else:
                        # 如果不是 'O' 生成 label2id 对应的标签
                        # e.g '全国青联委员经纪人' -> [0, 1, 2, 2, 2, 2, 2, 2, 2, 2] 注意：第一个 [CLS] 默认是 0
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    # 超过 maxlen 长度的部分被舍弃掉
                    break
            # 加上尾巴 [SEP]:102
            token_ids += [tokenizer._token_end_id]
            # 尾巴的 label 也是 0
            labels += [0]
            # 生成 segment_ids ，NER 只用了一个句子，所以这里都是 0
            segment_ids = [0] * len(token_ids)
            # 把样本组装成 batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            # 如果凑齐了 batchsize 个，或者到 epoch 的最后一个，那么就返回。
            if len(batch_token_ids) == self.batch_size or is_end:
                # 进行 padding 操作
                # [bs,seq] 默认按照 bs 中最大的长度进行填充，保证每个 bs 的长度是一致，用 0 填充
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []



class NamedEntityRecognizer(ViterbiDecoder):
    """
    命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text)) # 预测
        T = set([tuple(i) for i in d if i[1] != 'O']) #真实
        X += len(R & T) 
        Y += len(R) 
        Z += len(T)
    precision, recall =  X / Y, X / Z
    f1 = 2*precision*recall/(precision+recall)
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self,valid_data):
        self.best_val_f1 = 0
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
#         print(NER.trans)
        f1, precision, recall = evaluate(self.valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('../model/best_model_epoch_10.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )



evaluator = Evaluator(valid_data)
train_generator = data_generator(train_data, batch_size)

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    callbacks=[evaluator]
)

