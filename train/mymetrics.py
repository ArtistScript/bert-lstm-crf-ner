#coding:utf-8
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
def compute(prediction,label_ids,label_list):
    """

    :param prediction: [batch_size, sequence_length]
    :param label_ids:  [batch_size, sequence_length]
    :param label_list: 参数传入，list or set
    :return:
    """
    prediction=np.array(prediction)
    label_ids=np.array(label_ids)
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):  # i从1开始增加
        label_map[label] = i
    id2label = {value: key for key, value in label_map.items()}
    names=['I-LOC', 'I-ORG', 'B-ORG', 'B-PER','I-PER', 'B-LOC']
    nameids=[label_map[n] for n in names]
    for nid in nameids:
        pred=(prediction==nid)*1
        trues=(label_ids==nid)*1
        print('accuracy: '+id2label[nid]+" %.2f"%(accuracy_score(trues,pred)))#真实值写在左边
        pass
    pass