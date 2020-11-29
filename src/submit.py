import tensorflow as tf
import bert4keras
import keras
import pandas as pd
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

from glob import glob
test_files = glob("../data/test_data/*.txt")

# 预训练模型的超参数
maxlen = 300
epochs = 15
batch_size = 8
bert_layers = 12
learing_rate = 1e-5 
crf_lr_multiplier = 1000 
rnn_lr_multiplier = 1000
# 数据处理参数
symbol = ['？','⋯','…','﹗']
max_sent_length = 250
max_input_length = 300     
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
model.summary()

model_path = '../model/best_model_epoch_10.weights'
model.load_weights(model_path)

class NamedEntityRecognizer(ViterbiDecoder):
    """
    命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        nodes = model.predict([[token_ids], [segment_ids]])[0]
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

def test_predict(data, NER_):
    test_ner =[]
    for text in tqdm(data):
        cut_text_list, cut_index_list = utils.agg_sent([text],symbol, max_sent_length, max_input_length)
        posit = 0
        item_ner = []
        index =1
        for str_ in cut_text_list:
            ner_res  = NER_.recognize(str_)
            for tn in ner_res:
                ans = {}
                ans["label_type"] = tn[1]
                ans['index'] = str(index)
                ans["start_pos"] = text.find(tn[0],posit)
                ans["end_pos"] = ans["start_pos"] + len(tn[0])-1
                posit = ans["end_pos"]
                ans["res"] = tn[0]
                item_ner.append(ans)
                index +=1
        test_ner.append(item_ner)
    
    return test_ner

df_ret = {'ID':[],'Category':[],'Pos_b':[],'Pos_e':[],'Privacy':[]}
for file in test_files:
    with open(file, "r", encoding="utf-8") as f:
        line = f.read()
        line = [line]
        ret = test_predict(line, NER)
    idx = os.path.basename(file).split('.')[0]
    for line in ret[0]:
        df_ret['ID'].append(idx)
        df_ret['Category'].append(line['label_type'])
        df_ret['Pos_b'].append(line['start_pos'])
        df_ret['Pos_e'].append(line['end_pos'])
        df_ret['Privacy'].append(line['res'])

version = '20201118'
df_ret_ = pd.DataFrame(df_ret)
df_ret_ = df_ret_.sort_values('ID')
df_ret_.to_csv('../submit/predict{}.csv'.format(version),index=None)

file_name = '../submit/predict{}.csv'.format(version)
utils.checkout(file_name)