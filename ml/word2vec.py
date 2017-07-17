# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# from gensim import corpora
#
# documents = ["Human machine interface for lab abc computer applications",
#              "A survey of user opinion of computer system response time",
#              "The EPS user interface management system",
#              "System and human system engineering testing of EPS",
#              "Relation of user perceived response time to error measurement",
#              "The generation of random binary unordered trees",
#              "The intersection graph of paths in trees",
#              "Graph minors IV Widths of trees and well quasi ordering",
#              "Graph minors A survey" ]
#
# stoplist = set('for a of the and to in'.split())
# texts = [ [word for word in doc.lower().split() if word not in stoplist] for doc in documents]
#
# from collections import defaultdict
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1
#
# texts = [[token for token in text if frequency[token] > 1] for text in texts]
#
# from pprint import pprint
# pprint(texts)
#
# dictionary = corpora.Dictionary(texts)
# dictionary.save('/tmp/deerwester.dict')
# print(dictionary)
# pprint(dictionary.token2id)
# new_doc = 'human computer interaction minors human minors'
# new_vec = dictionary.doc2bow(new_doc.lower().split())
# pprint(new_vec)


import gensim,logging
from pprint import pprint
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = [['first', 'sentence'],['second','sentence']]
model = gensim.models.Word2Vec(sentences,min_count=1)
pprint(model)
pprint(model.wv.vocab)

