import numpy as np
import sys

if __name__ == '__main__':
    m = np.loadtxt("m.txt")
    #print(m)
    b = m[:, -1]
    #print(b)
    a = np.delete(m, -1, 1)
    #print(a)
    x = np.loadtxt(sys.argv[1])
    #x = np.loadtxt("D:\\x.txt")
    x_vec = x.T
    #print(x_vec)
    expected_b = np.dot(a, x.T)
    #print(expected_b)
    
    fl = 0
    for i in range(len(b)):
        if (abs(b[i] - expected_b[i]) > 0.001):
            print("Error in x ", i)
            fl = 1
    if (fl == 0):
        print("Correct!")
  
