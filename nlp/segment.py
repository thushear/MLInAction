import jieba
import jieba.analyse as analyse
import jieba.posseg as pesg

lines = open('jieba.txt', encoding='utf-8').read()
linestr = ''.join(tuple(lines))
# 关键词提取 tf-idf
tags = analyse.extract_tags(linestr, topK=10, withWeight=True)

print(tags)

print(analyse.textrank(linestr, topK=10, withWeight=True))

jieba.add_word('石墨烯')
jieba.add_word('凯瑟琳')
jieba.load_userdict('userdict.txt')

test_sent = ("李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
"例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
"「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。")

words = jieba.cut(test_sent)
print('/'.join(words))
 
result = pesg.cut(test_sent)
for w in result:
    print(w.word,'/',w.flag,',',end=' ')



