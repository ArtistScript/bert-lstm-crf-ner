#coding:utf-8
import re
import json
# def parse_tag(t):
#     m = re.match(r'^([^-]*)-(.*)$', t)
#     return m.groups() if m else (t, '')
#
# a="B-LOC"
# print(parse_tag(a))
labels='O O O O O O O B-LOC I-LOC O B-LOC I-LOC O O O O O O'
tokens='海 钓 比 赛 地 点 在 厦 门 与 金 门 之 间 的 海 域 。'
print(len(labels.split()))
print(len(tokens.split()))