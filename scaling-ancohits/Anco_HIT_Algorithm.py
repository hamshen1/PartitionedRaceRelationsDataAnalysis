import pickle
import sys
#import IPython
import numpy as np
#import pdb
from scipy.sparse import rand
#from itertools import izip as zip
'''
import logging
logging.basicConfig(level=logging.DEBUG)
l = logging.getLogger(name='cohit')
l.setLevel(logging.DEBUG)
'''
#from numpy.matrix import transpose

# define epsilon here!
epsilon = 0.0000000001

def unpickler(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data


def pickler(var, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(var, f)


def ANCOHITS(A, epsilon):
    maxIter = 100000
    #length = A.size
    length = np.size(A, 0)
    #s1 = np.full((length,1), 1)
    s1 = rand(length, 1, density=1, format='csr').toarray()

    A_T = np.transpose(A)
    #realmin = np.finfo(float).tiny
    realmin = 0.2

    for i in range(maxIter):
        s1old = s1
        #pdb.set_trace()
        #l.debug(np.size(s1,0))
        T1 = np.abs(A_T).dot(np.abs(s1))
        divider_1 = np.add(T1, realmin)
        s2 = np.divide(A_T.dot(s1), divider_1)


        T2 = np.abs(A).dot(np.abs(s2))
        divider_2 = np.add(T2 ,realmin)
        s1 = np.divide(A.dot(s2), divider_2)

        diff = np.linalg.norm(np.subtract(s1, s1old))
        #l.debug(diff)

        if abs(diff) <= epsilon:
            break

    return s1, s2


if __name__ == '__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        data = unpickler(fname)
        s1, s2 = ANCOHITS(data, epsilon)
        pickler(s1, 's1')
        pickler(s2, 's2')
        np.savetxt("s1.txt", s1, fmt="%s")
        np.savetxt("s2.txt", s2, fmt="%s")

        #IPython.embed()
    else:
        print('Usage: %s file_name'% sys.argv[0])
