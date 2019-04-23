from sklearn.metrics import accuracy_score, precision_score, recall_score
def compute(prediction,label_ids,label_list):
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):  # i从1开始增加
        label_map[label] = i
    id2label = {value: key for key, value in label_map.items()}

    pass