# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum

import codecs
import collections
import json
import re
import sys

import six
import modeling
import tokenization
import tensorflow as tf
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.estimator import Estimator


class Sentence2VecConfig(object):
  """Configuration for `Sentence2Vec`."""
  
  def __init__(self,
               vocab_file="",
               bert_config_file="",
               init_checkpoint="",
               use_one_hot_embeddings=False,
               feature_pooling_layer=-2,
               max_seq_length=128,
               batch_size=8,
               do_lower_case=False):
    """
    vocab_file--词表, bert_config_file--模型配置文件
    init_checkpoint--模型, feature_pooling_layer-特征抽取层
    max_seq_length--句子最大长度, batch-size--批大小
    """
    self.vocab_file = vocab_file
    self.bert_config_file = bert_config_file
    self.init_checkpoint = init_checkpoint
    self.use_one_hot_embeddings = use_one_hot_embeddings
    self.feature_pooling_layer = feature_pooling_layer
    self.max_seq_length = max_seq_length
    self.batch_size = batch_size
    self.do_lower_case = do_lower_case

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `Sentence2Vec` from a Python dictionary of parameters."""
    config = Sentence2VecConfig(do_lower_case=True)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `Sentence2Vec` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = json.loads(reader.read())
    return cls.from_dict(text)

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

class PoolingStrategy(Enum):
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # corresponds to [CLS] for single sequences
    LAST_TOKEN = 5  # corresponds to [SEP] for single sequences
    CLS_TOKEN = 4  # corresponds to the first token for single seq.
    SEP_TOKEN = 5  # corresponds to the last token for single seq.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()

class SentenceVecTuple(collections.namedtuple("SentenceVecTuple", ("unique_id", "sentence", "vector"))):
    pass

class Sentence2Vec(object):
    def __init__(self, sen2vec_conf_file):
        tf.logging.set_verbosity(tf.logging.WARN)
        ##Sentence2Vec配置
        self.sen2vec_conf =  Sentence2VecConfig.from_json_file(sen2vec_conf_file)
        ##bert模型+分词器
        self.bert_config = modeling.BertConfig.from_json_file(self.sen2vec_conf.bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.sen2vec_conf.vocab_file, do_lower_case=self.sen2vec_conf.do_lower_case)

        self.model_fn = self.model_fn_builder(
            bert_config=self.bert_config,
            init_checkpoint=self.sen2vec_conf.init_checkpoint,
            use_one_hot_embeddings=self.sen2vec_conf.use_one_hot_embeddings,
            pooling_layer=self.sen2vec_conf.feature_pooling_layer)

        # Remove TPU Estimator.
        #estimator = tf.contrib.tpu.TPUEstimator(
        #    use_tpu=self.sen2vec_conf.use_tpu,
        #    model_fn=model_fn,
        #    config=run_config,
        #    predict_batch_size=self.sen2vec_conf.batch_size)
        self.params = {}
        self.params["batch_size"] = self.sen2vec_conf.batch_size
        self.estimator = Estimator(self.model_fn, params=self.params)
        self.seq_length = self.sen2vec_conf.max_seq_length

    def run(self, sentence_list):
        """
        批量句子计算sen2vec
        @return []按照顺序
        """
        rsl = []
        features = [self.sentence2feature(sentence_list[i], i, self.seq_length) for i in range(len(sentence_list))]
        input_fn = self.input_fn_builder(features, self.seq_length)
        for result in self.estimator.predict(input_fn, yield_single_examples=True):
            unique_id = int(result["unique_id"])
            encodes = [round(float(x), 6) for x in result["encodes"].flat]
            rsl.append(SentenceVecTuple(unique_id=unique_id,
                                           sentence=sentence_list[unique_id],
                                           vector=encodes))
        return rsl

    def model_fn_builder(self, bert_config, init_checkpoint, use_one_hot_embeddings = False,
                        pooling_strategy=PoolingStrategy.REDUCE_MEAN,
                        pooling_layer=-2):
      """Returns `model_fn` closure for Estimator."""
    
      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
    
        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]
    
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)
    
        if mode != tf.estimator.ModeKeys.PREDICT:
          raise ValueError("Only PREDICT modes are supported: %s" % (mode))
    
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
          init_string = ""
          if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
          tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                          init_string)
    
        #all_layers = model.get_all_encoder_layers()
        encoder_layer = model.all_encoder_layers[pooling_layer]
        if pooling_strategy == PoolingStrategy.REDUCE_MEAN:
            pooled = tf.reduce_mean(encoder_layer, axis=1)
        elif pooling_strategy == PoolingStrategy.REDUCE_MAX:
            pooled = tf.reduce_max(encoder_layer, axis=1)
        elif pooling_strategy == PoolingStrategy.REDUCE_MEAN_MAX:
            pooled = tf.concat([tf.reduce_max(encoder_layer, axis=1), tf.reduce_max(encoder_layer, axis=1)], axis=1)
        elif pooling_strategy == PoolingStrategy.FIRST_TOKEN or pooling_strategy == PoolingStrategy.CLS_TOKEN:
            pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
        elif pooling_strategy == PoolingStrategy.LAST_TOKEN or pooling_strategy == PoolingStrategy.SEP_TOKEN:
            seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
            rng = tf.range(0, tf.shape(seq_len)[0])
            indexes = tf.stack([rng, seq_len - 1], 1)
            pooled = tf.gather_nd(encoder_layer, indexes)
        else:
            raise NotImplementedError() 
        predictions = {
            "unique_id": unique_ids,
            "encodes": pooled
        }
        
        return EstimatorSpec(mode=mode, predictions=predictions)
    
      return model_fn

    def input_fn_builder(self, features, seq_length):
      """Creates an `input_fn` closure to be passed to Estimator."""
  
      all_unique_ids = []
      all_input_ids = []
      all_input_mask = []
      all_input_type_ids = []
    
      for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)
    
      def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
    
        num_examples = len(features)
    
        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_type_ids":
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })
    
        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d
    
      return input_fn
    
    def sentence2feature(self, sentence, unique_id, seq_length):
      line = tokenization.convert_to_unicode(sentence)
      assert line
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      example = InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
      tokens_a = self.tokenizer.tokenize(example.text_a)
      tokens_b = self.tokenizer.tokenize(example.text_b) if example.text_b else None
      if tokens_b:
          # Modifies `tokens_a` and `tokens_b` in place so that the total
          # length is less than the specified length.
          # Account for [CLS], [SEP], [SEP] with "- 3"
        self._truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
      else:
          # Account for [CLS] and [SEP] with "- 2"
          if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

      feature = self.tokens2feature(example.unique_id, tokens_a, tokens_b, seq_length)
      return feature
        
    def tokens2feature(self, unique_id, tokens_a, tokens_b, seq_length):
      """
      seq_length 
      """
      tokens = []
      input_type_ids = []
      
      ##Q Part=a
      tokens.append("[CLS]")
      input_type_ids.append(0)
      for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
      tokens.append("[SEP]")
      input_type_ids.append(0)
      ##A Part=b
      if tokens_b:
        for token in tokens_b:
          tokens.append(token)
          input_type_ids.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(1)
      input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
      input_mask = [1] * len(input_ids)
      # Zero-pad up to the sequence length.
      while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)
    
      assert len(input_ids) == seq_length
      assert len(input_mask) == seq_length
      assert len(input_type_ids) == seq_length
      return InputFeatures(unique_id=unique_id, tokens=tokens, input_ids=input_ids, input_mask=input_mask, input_type_ids=input_type_ids)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
      """Truncates a sequence pair in place to the maximum length."""
      while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
          break
        if len(tokens_a) > len(tokens_b):
          tokens_a.pop()
        else:
          tokens_b.pop()

if __name__ == "__main__":
    sen2vec = Sentence2Vec(sys.argv[1])
    #批量传入句子，传出句向量
    result = sen2vec.run(["红色", "大红色"])
    for tp in result:
            print(tp)
