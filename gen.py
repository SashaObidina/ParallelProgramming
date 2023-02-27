import numpy as np
import sys

N1 = int(sys.argv[1])

data1 = np.random.randint(-10, 10, size=(N1, N1+1))

np.savetxt('m.txt', data1, fmt='%10.0f')
