from sklearn import datasets
iris = datasets.load_iris()
print(iris.data[:5])
print(iris.target[:5])

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print('='*40)
print(iris.target)
print('='*40)
print(y_pred)
right_num = (iris.target == y_pred).sum()
print('total testing num :%d,naive bayes accurary:%f'%(iris.data.shape[0],float(right_num)/iris.data.shape[0]))

#总共有9种新闻类别，我们给每个类别一个编号
lables = ['A','B','C','D','E','F','G','H','I']
import random,collections,math
def shuffle(inFile):
    textLines = [line.strip() for line in  open(inFile)]
    print('正在准备训练和测试数据 请稍后')
    random.shuffle(textLines)
    num = len(textLines)
    trainText = textLines[:3*num/5]
    testText = textLines[3*num/5:]
    print('准备测试数据和训练数据完毕')
    return trainText, testText

def doc_dict():
    return [0]*len(lables)

def label2id(label):
    for i in range(len(lables)):
        if label == lables[i]:
            return i
    raise Exception('error lable %s' % (label))

def mutual_info(N,Nij,Ni_,N_j):
    '''
        计算互信息，这里log的底取为2
    '''
    return Nij * 1.0 / N * math.log(N * (Nij+1)*1.0/(Ni_*N_j))/ math.log(2)

def count_for_cates(trainText,featureFile):
    docCount = [0] * len(lables)
    wordCount = collections.defaultdict(doc_dict())
    for line in trainText:
        label,text = line.strip().split(' ',1)
        index = label2id(label[0])
        words = text.split(' ')
        for word in words:
            wordCount[word][index] += 1
            docCount[index]+=1
    print('计算互信息 提取关键特征词中 ')
    miDict = collections.defaultdict(doc_dict())
    N = sum(docCount)
    for k , vs in wordCount.items():
        for i in range(len(vs)):
            N11 = vs[i]
            N10 = sum(vs) - N11
            N01 = docCount[i] - N11
            N00 = N - N11 - N10 - N01
            mi = mutual_info(N, N11, N10 + N11, N01 + N11) + mutual_info(N, N10, N10 + N11, N00 + N10) + mutual_info(N,
                                                                                                                     N01,
                                                                                                                     N01 + N11,
                                                                                                                     N01 + N00) + mutual_info(
                N, N00, N00 + N10, N00 + N01)
            miDict[k][i] = mi
    fWords = set()
    for i in range(len(docCount)):
        keyf = lambda x: x[1][i]
        sortedDict = sorted(miDict.items(), key=keyf, reverse=True)
        for j in range(100):
            fWords.add(sortedDict[j][0])
    out = open(featureFile, 'w')
    # 输出各个类的文档数目
    out.write(str(docCount) + "\n")
    # 输出互信息最高的词作为特征词
    for fword in fWords:
        out.write(fword + "\n")
    print
    "特征词写入完毕..."
    out.close()
