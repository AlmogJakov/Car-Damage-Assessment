import pickle

import timeit
start = timeit.default_timer()

vectors = []
# load the vectors from binary file
with open('database/vectors.pkl', 'rb') as f:
    vectors = pickle.load(f)

# print(vectors)
print(len(vectors))
stop = timeit.default_timer()
print('Time: ', stop - start)