from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf


data = Dataset.load_builtin('ml-100k')
print(data)