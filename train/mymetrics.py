from sklearn.metrics import accuracy_score, precision_score, recall_score
def compute(prediction,label_ids,label_list):
    """

    :param prediction: [batch_size, sequence_length]
    :param label_ids:  [batch_size, sequence_length]
    :param label_list: 参数传入，list or set
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):  # i从1开始增加
        label_map[label] = i
    id2label = {value: key for key, value in label_map.items()}
    names=['I-LOC', 'I-ORG', 'B-ORG', 'B-PER','I-PER', 'B-LOC']
    nameid=[label_map[n] for n in names]

    pass