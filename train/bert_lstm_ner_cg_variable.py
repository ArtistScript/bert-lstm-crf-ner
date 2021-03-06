#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
reference from :zhoukaiyin/

@Author:Macan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
#sys.path.append('/home/idm/dzt/models/nlp_model/bert-lstm-crf-ner/')
import collections
import os
import numpy as np
import tensorflow as tf
import codecs
import pickle

from train import tf_metrics
from bert import modeling
from bert import optimization
from bert import tokenization
from train import mymetrics
# import

from train.models import create_model, InputFeatures, InputExample

__version__ = '0.1.0'

__all__ = ['__version__', 'DataProcessor', 'NerProcessor', 'write_tokens', 'convert_single_example',
           'filed_based_convert_examples_to_features', 'file_based_input_fn_builder',
           'model_fn_builder', 'train']



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines


class NerProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, labels=None):
        #传入参数可能是labels文件路径，也可能是逗号分隔的labels文本
        if labels is not None:
            try:
                # 支持从文件中读取标签类型
                if os.path.exists(labels) and os.path.isfile(labels):
                    with codecs.open(labels, 'r', encoding='utf-8') as fd:
                        for line in fd:
                            self.labels.append(line.strip())
                else:
                    # 否则通过传入的参数，按照逗号分割
                    self.labels = labels.split(',')
                self.labels = set(self.labels) # to set
            except Exception as e:
                print(e)
        # 通过读取train文件获取标签的方法会出现一定的风险。
        if os.path.exists(os.path.join(self.output_dir, 'label_list.pkl')):
            with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'rb') as rf:
                self.labels = pickle.load(rf)
        else:
            if len(self.labels) > 0:
                #pkl文件不存在，就按照读取的标签集合加上一些其他标签，写入pkl
                self.labels = self.labels.union(set(["X", "[CLS]", "[SEP]"]))
                with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'wb') as rf:
                    pickle.dump(self.labels, rf)
            else:
                #如果什么都没有，都按照代码写好的标签集合
                self.labels = ["O", 'B-TIM', 'I-TIM', "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        return self.labels

    def _create_example(self, lines, set_type):
        #比如set_type是train，就表示训练数据
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            #使用bert内部的tokenization包，把字符串转化成unicode
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            # if i == 0:
            #     print('label: ', label)
            #模型训练的输入类，guid为唯一数据id
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                self.labels.add(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            return lines


def write_tokens(tokens, output_dir, mode):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1): #i从1开始增加
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    #因为从训练数据读取后，字，标签标记，都是用空格分隔的
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # 一般不会出现else，因为只有一个词
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []  #segment_ids的作用是？？？
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过句首和句尾使用不同的标志来标注，使用CLS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    #调用bert内部的token2id函数，把token转化成bert使用的tokenid
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)   #mask是隐藏的token，用于模型训练
    # label_mask = [1] * len(input_ids)
    # padding, 使用
    #小于序列长度的，进行补全操作
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类，使用自定义类保存训练index化数据，
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # mode='test'的时候才有效，把token写入文件
    write_tokens(ntokens, output_dir, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
        # 返回的是bert模型训练需要的index化token，label，mask，segment等信息
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()#除了记录k,v，还会记录k放入的顺序
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        # Example中存放features特征，放入example是为了便于把特征序列化存储
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder,batch_size):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        #把recod变成字典example？,可能写入，读取都是按照tensorflow的某个标准
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    d = tf.data.TFRecordDataset(input_file)
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=300)
    #通过map函数，调用_decode_record，把int64的数据转化成int32的数据，通过apply，把数据转化成batch的形式
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=4,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                       # num_parallel_calls=tf.data.experimental.AUTOTUNE, #根据机器动态调整并行数
                                                       drop_remainder=drop_remainder))
    else:
        d = d.shuffle(buffer_size=300)
        # 通过map函数，调用_decode_record，把int64的数据转化成int32的数据，通过apply，把数据转化成batch的形式
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=1000,
                                                       num_parallel_calls=4,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                       # num_parallel_calls=tf.data.experimental.AUTOTUNE, #根据机器动态调整并行数
                                                       drop_remainder=drop_remainder))
    d = d.prefetch(buffer_size=4)

    return d


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, args):
    """
    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_tpu:
    :param use_one_hot_embeddings:
    :return:
    """

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print('shape of input_ids', input_ids.shape)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        #全部损失，分数，，预测类别
        total_loss, logits, trans, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)
        # tf.summary.scalar('total_loss', total_loss)
        # tf.summary.scalar('logits',logits)
        # tf.summary.scalar('trans',trans)
        # tf.summary.scalar('pred_ids',pred_ids)
        #所有需要训练的变量
        tvars = tf.trainable_variables()
        # 加载BERT模型，assignmen_map，加载的预训练变量值
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                 modeling.get_assignment_map_from_checkpoint(tvars,
                                                             init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # 打印变量名
        # logger.info("**** Trainable Variables ****")
        #
        # # 打印加载模型的参数
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     logger.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            #train_op = optimizer.optimizer(total_loss, learning_rate, num_train_steps)
            train_op = optimization.create_optimizer(
                 total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            hook_dict = {}
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.save_summary_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, pred_ids):
                return {
                    "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
                }

            eval_metrics = metric_fn(label_ids, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec

    return model_fn


# def load_data():
#     processer = NerProcessor()
#     processer.get_labels()
#     example = processer.get_train_examples(FLAGS.data_dir)
#     print()

def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        tf.logging.info('checkpoint file not exits:'.format(os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last


def adam_filter(model_path):
    """
    去掉模型中的Adam相关参数，这些参数在测试的时候是没有用的
    :param model_path:
    :return:
    """
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, 'model.ckpt'))


def train(args):
    print('using bert_lstm_ner_cg_variable')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

    #一个处理的类，包括训练数据的输入等
    processors = {
        "ner": NerProcessor
    }
    #载入bert配置文件
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

    #检查序列的最大长度是否超出范围
    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if args.clean and args.do_train:
        if os.path.exists(args.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)

    #check output dir exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #通过output_dir初始化数据处理类，processor
    processor = processors[args.ner](args.output_dir)
    #通过bert字典，初始化bert自带分词类
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)


    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None

    #一般都是True
    if args.do_train and args.do_eval:
        # 加载训练数据,train和dev，会自动拼接文件夹和train.txt
        #返回的训练数据是一个list，每个元素是两个字符串，空格分隔字，空格分隔字标记，并写入训练examples类中
        train_examples = processor.get_train_examples(args.data_dir)
        #训练步数
        num_train_steps = int(
            len(train_examples) *1.0 / args.batch_size * args.num_train_epochs)
        if num_train_steps < 1:
            raise AttributeError('training data is so small...')
        #
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", args.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        #读取验证集
        eval_examples = processor.get_dev_examples(args.data_dir)

        # 打印验证集数据信息
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", args.batch_size)

    #获取标签集合，是一个list，如果是自己输入的话，这里一定不能搞错，会影响最后预测的类目
    #一般label_list为所以的标签，[CLS],[SEP],O，这三个
    label_list = processor.get_labels()
    # label_list=["O", 'B-TIM', 'I-TIM', "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
    # label_list=['B-PER', '[SEP]', 'I-ORG', 'O', 'I-LOC', 'I-PER', 'B-ORG', 'B-LOC', '[CLS]', 'X']
    prod_map = {"C材质": "C", "R人群": "R", "K口味": "K", "G功能呢": "G", "K款式": "K", "X剂型": "J", "S款式": "K", "J剂型": "J",
                "N年龄": "N",
                "G功能": "G", "Z品牌": "Z", "E品牌": "Z", "M明星": "M", "D地域": "D", "P品类": "P", "V规格": "V", "F风格": "F",
                "Y颜色": "Y", "J季节": "S"}
    label_list = ["C","R","K","X","G", "J", "N","M","Z","D","S","P","F", "Y","V","[CLS]", "[SEP]","O"]
    num_labels = len(label_list) + 1
    init_checkpoint = args.init_checkpoint
    learning_rate = args.learning_rate

    with tf.name_scope('input'):
        input_ids = tf.placeholder(tf.int32, [None, args.max_seq_length])
        input_mask = tf.placeholder(tf.int32, [None, args.max_seq_length])
        segment_ids  = tf.placeholder(tf.int32, [None, args.max_seq_length])
        label_ids = tf.placeholder(tf.int32, [None, args.max_seq_length])
        # is_training=tf.placeholder(tf.bool)
        #对参数赋值，对于训练模型来说

    #is_training全部变成false，不使用dropout层
    total_loss, logits, trans, pred_ids = create_model(
        bert_config, False, input_ids, input_mask, segment_ids, label_ids,
        num_labels, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)
    # total_loss_evl, logits_evl, trans_evl, pred_ids_evl = create_model(
    #     bert_config, False, input_ids, input_mask, segment_ids, label_ids,
    #     num_labels, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)
    accuracy, acc_op = tf.metrics.accuracy(labels=label_ids,predictions=pred_ids)   # 计算准确率,pred_ids是预测序列，
    # accuracy_evl, acc_op_evl = tf.metrics.accuracy(labels=label_ids, predictions=pred_ids_evl)  # 计算准确率,pred_ids是预测序列
    #输出loss的smmary
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('accuracy', acc_op)
    #---------------------输出验证集，测试集数据------------------------------
    # is_training_evl = False #bert模型不采用training模式
    # total_loss_evl, logits_evl, trans_evl, pred_ids_evl = create_model(
    #     bert_config, is_training_evl, input_ids, input_mask, segment_ids, label_ids,
    #     num_labels, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)
    # accuracy_evl, acc_op_evl = tf.metrics.accuracy(labels=label_ids, predictions=pred_ids_evl)  # 计算准确率,pred_ids是预测序列
    # tf.summary.scalar('total_loss_evl', total_loss_evl)
    # tf.summary.scalar('accuracy_evl', acc_op_evl)
    #----------------------------------------------------------------------------
    #加载预训练隐变量
    tvars = tf.trainable_variables()
    # 加载BERT模型，assignmen_map，加载的预训练变量值
    if init_checkpoint:  #只会运行一次
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars,
                                                        init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    #优化loss
    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

    # 1. 将数据转化为tf_record 数据,并把训练数据序列化，并写出到文件
    train_file = os.path.join(args.output_dir, "train.tf_record")
    #ok
    if not os.path.exists(train_file):
        filed_based_convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, train_file, args.output_dir)

    # 2.读取record 数据，组成batch，把上一部输出到文件的训练数据读取
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=args.max_seq_length,
        is_training=True,
        drop_remainder=True,
        batch_size=args.batch_size)
    # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    eval_file = os.path.join(args.output_dir, "eval.tf_record")
    if not os.path.exists(eval_file):
        filed_based_convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, eval_file, args.output_dir)
    #构建验证集数据
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=args.max_seq_length,
        is_training=False,
        drop_remainder=False,
        batch_size=args.batch_size)
    #构建预测集数据
    predict_examples = processor.get_test_examples(args.data_dir)
    predict_file = os.path.join(args.output_dir, "predict.tf_record")
    #保存训练数据到本地
    filed_based_convert_examples_to_features(predict_examples, label_list,
                                             args.max_seq_length, tokenizer,
                                             predict_file, args.output_dir, mode="test")
    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=args.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder,
        batch_size=args.batch_size)

    train_input=train_input_fn.make_one_shot_iterator()
    eval_input=eval_input_fn.make_one_shot_iterator()
    predict_input=predict_input_fn.make_one_shot_iterator()
    # sess = tf.InteractiveSession()
    max_step=2000
    merged = tf.summary.merge_all()

    meta_train_data = train_input.get_next()
    meta_eval_data = eval_input.get_next() #获取验证数据集
    meta_predict_data=predict_input.get_next()
    #参数batch_size是64，train_batch_size是32，不知道train_batch_size是什么用的
    #------------------解决FailedPreconditionError:问题，初始化所有变量，不知道这样会不会影响初始化的bert预训练变量------------------
    # init_op = tf.initialize_all_variables()
    # init_global= tf.global_variables_initializer()
    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allow_growth = True  # 动态申请显存
    sess = tf.Session(config=config)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./log_predict', sess.graph)
    eval_data = sess.run(meta_eval_data)
    predict_data=sess.run(meta_predict_data)
    print(label_list)
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    print(label_map)
    #------------------问题--------------------------------------------------------------------------------------------

    for i in range(max_step):

        #把tensor转化为numpy输入
        train_data=sess.run(meta_train_data)
        #is_traing,是否使用bert，以及lstm中的dropout层
        #istraing，True，False混合使用，会涉及共享变量的问题，貌似共享变量后产生bug，暂时先false，不使用dropout。
        #好像后面的variable_scope都reuse，也不会产生问题
        sess.run(train_op,feed_dict={input_ids:train_data['input_ids'],input_mask:train_data['input_mask'],
                                     segment_ids:train_data['segment_ids'],label_ids:train_data['label_ids']})
        if i%10==1:
            train_summary,acco, prediction = sess.run([merged,acc_op,pred_ids], feed_dict={input_ids:train_data['input_ids'],input_mask:train_data['input_mask'],
                                     segment_ids:train_data['segment_ids'],label_ids:train_data['label_ids']})
            # acco_evl,prediction_eval=sess.run([acc_op_evl,pred_ids_evl],feed_dict={input_ids:eval_data['input_ids'],input_mask:eval_data['input_mask'],
            #                          segment_ids:eval_data['segment_ids'],label_ids:eval_data['label_ids']})
            #预测训练集准确率，看下变量重用是否可行
            acco_evl,prediction_eval=sess.run([acc_op,pred_ids],feed_dict={input_ids:predict_data['input_ids'],input_mask:predict_data['input_mask'],
                                     segment_ids:predict_data['segment_ids'],label_ids:predict_data['label_ids']})
            train_writer.add_summary(train_summary, i)
            print('saving summary at %s, accuracy %s, accuracy_eval %s,length predict data %s'%(i,acco,acco_evl,len(predict_data['label_ids'])))
            # print(prediction)
            # print(train_data['label_ids'])
            mymetrics.compute(prediction_eval,predict_data['label_ids'],label_list)
    train_writer.close()


    # if args.do_predict:
    #     token_path = os.path.join(args.output_dir, "token_test.txt")
    #     if os.path.exists(token_path):
    #         os.remove(token_path)
    #
    #     with codecs.open(os.path.join(args.output_dir, 'label2id.pkl'), 'rb') as rf:
    #         label2id = pickle.load(rf)
    #         id2label = {value: key for key, value in label2id.items()}
    #
    #     predict_examples = processor.get_test_examples(args.data_dir)
    #     predict_file = os.path.join(args.output_dir, "predict.tf_record")
    #     #保存训练数据到本地
    #     filed_based_convert_examples_to_features(predict_examples, label_list,
    #                                              args.max_seq_length, tokenizer,
    #                                              predict_file, args.output_dir, mode="test")
    #
    #     tf.logging.info("***** Running prediction*****")
    #     tf.logging.info("  Num examples = %d", len(predict_examples))
    #     tf.logging.info("  Batch size = %d", args.batch_size)
    #
    #     predict_drop_remainder = False
    #     #从文件读取测试集特征
    #     predict_input_fn = file_based_input_fn_builder(
    #         input_file=predict_file,
    #         seq_length=args.max_seq_length,
    #         is_training=False,
    #         drop_remainder=predict_drop_remainder)
    #
    #     result = estimator.predict(input_fn=predict_input_fn)
    #     output_predict_file = os.path.join(args.output_dir, "label_test.txt")
    #
    #     #把结果也按照字，字标签这样输出
    #     def result_to_pair(writer):
    #         for predict_line, prediction in zip(predict_examples, result):
    #             idx = 0
    #             line = ''
    #             line_token = str(predict_line.text).split(' ')
    #             label_token = str(predict_line.label).split(' ')
    #             #序列长度
    #             len_seq = len(label_token)
    #             if len(line_token) != len(label_token):
    #                 tf.logging.info(predict_line.text)
    #                 tf.logging.info(predict_line.label)
    #                 break
    #             for id in prediction:
    #                 if idx >= len_seq:
    #                     break
    #                 if id == 0:
    #                     continue
    #                 curr_labels = id2label[id]
    #                 if curr_labels in ['[CLS]', '[SEP]']:
    #                     continue
    #                 try:
    #                     line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
    #                 except Exception as e:
    #                     tf.logging.info(e)
    #                     tf.logging.info(predict_line.text)
    #                     tf.logging.info(predict_line.label)
    #                     line = ''
    #                     break
    #                 idx += 1
    #             writer.write(line + '\n')
    #
    #     with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
    #         result_to_pair(writer)
    #     from train import conlleval
    #     eval_result = conlleval.return_report(output_predict_file)
    #     print(''.join(eval_result))
    #     # 写结果到文件中
    #     with codecs.open(os.path.join(args.output_dir, 'predict_score.txt'), 'a', encoding='utf-8') as fd:
    #         fd.write(''.join(eval_result))
    # filter model
    # if args.filter_adam_var:
    #     adam_filter(args.output_dir)

if __name__=='__main__':
    import os
    from train.train_helper import get_args_parser
    from train.bert_lstm_ner_cg_estimator import train

    args = get_args_parser()
    if True:
        import sys

        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    # print(args)
    operation_sys="linux"
    if operation_sys=="windows":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
        args.task_name="NER"
        args.do_train=True
        args.do_eval=True
        args.do_predict=True
        args.data_dir="D:/project/python_project/bert-lstm-crf-ner\data_demo"
        args.vocab_file="D:/project/python_project/bert-lstm-crf-ner/bert\chinese_L-12_H-768_A-12/vocab.txt"
        args.bert_config_file="D:/project/python_project/bert-lstm-crf-ner/bert/chinese_L-12_H-768_A-12/bert_config.json"
        args.init_checkpoint="D:/project/python_project/bert-lstm-crf-ner/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
        args.max_seq_length=128
        args.train_batch_size=32
        args.learning_rate=2e-5
        args.num_train_epochs=3.0
        args.output_dir="D:/project/python_project/bert-lstm-crf-ner/output"
    elif operation_sys=="linux":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
        args.task_name = "NER"
        args.do_train = True
        args.do_eval = True
        args.do_predict = True
        args.data_dir = "/home/idm/dzt/kaola-ner/data_demo"
        args.vocab_file = "/home/idm/dzt/kaola-ner/chinese_L-12_H-768_A-12/vocab.txt"
        args.bert_config_file = "/home/idm/dzt/kaola-ner/chinese_L-12_H-768_A-12/bert_config.json"
        args.init_checkpoint = "/home/idm/dzt/kaola-ner/chinese_L-12_H-768_A-12/bert_model.ckpt"
        args.max_seq_length = 128
        args.train_batch_size = 32
        args.learning_rate = 2e-5
        args.num_train_epochs = 3.0
        args.output_dir = "./output"
    else:
        print("Please input the ")
    train(args=args)
