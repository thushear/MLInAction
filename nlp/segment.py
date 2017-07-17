import jieba
import jieba.analyse as analyse

lines = open('jieba.txt', encoding='utf-8').read()
linestr = ''.join(tuple(lines))

tags = analyse.extract_tags(linestr, topK=10, withWeight=True)

print(tags)

print(analyse.textrank(linestr, topK=10, withWeight=True))