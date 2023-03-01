from mpi4py import MPI
import time
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

size = 10
mtrxA = np.array([])

t = np.array([])
t = [0] * (size+2)
vecX = np.array([])
vecX = [0] * (size)

def elimination(data, r):
    #print("rank", r)
    start = int(data[-1][size+1])
    for k in range(len(data)-1):
        scaling = data[k][start] / data[-1][start]
        for l in range(start, size):
            data[k][l] -= data[-1][l] * scaling
        data[k][size] -= scaling * data[-1][size]
    data = np.delete(data, -1, 0)
    #print("after elim", data)
    return data 
       
def init_matrice():
    global mtrxA
    global B
    mtrxA = np.loadtxt("m.txt")
    #print(mtrxA)

if __name__ == '__main__':
    if (rank == 0):
        init_matrice()

        start_time = time.time()
                       
        for row_n in range(size - 1):
            newA = np.array([])
            data = np.array([])
            data_add = mtrxA[0:row_n+1][:]
            rows_number_add = np.array([]) #строки матрицы до row_n включительно
            
            for i in range(row_n+1):
                rows_number_add = np.append(rows_number_add, row_n)
            data_add = np.hstack((data_add, np.atleast_2d(rows_number_add).T))
            
            numpr = numprocs #номер процесса, начиная с которого data будет пустым
            
            for num in range(numprocs):
                data = mtrxA[row_n+num+1:size:numprocs][:]
                data = np.vstack([data, mtrxA[row_n][:]]) #добавили строку row_n
                if (len(data) > 1): #если есть данные, которые надо изменять
                    rows_number = np.array([]) #номера строк в data (доп.столбец)
                    for i in range(len(data) - 1):
                        rows_number = np.append(rows_number, (row_n+num+1)+i*numprocs)
                    rows_number = np.append(rows_number, row_n)
                    #print("newdata", data)
                    data = np.hstack((data, np.atleast_2d(rows_number).T))
                else: 
                    if (numpr == numprocs):
                        numpr = num
                    
                #print("data", data)
                
                if (num > 0):
                    comm.send(data, dest=num, tag = row_n)
                else: #работа 0-го процесса
                    if (len(data) > 1):
                        new_data = elimination(data, rank)
                        newA = new_data  
                      
            for n in range(1, numpr):
                new_data = comm.recv(source=n, tag=n)
                newA = np.append(newA, new_data, 0)
            newA = np.append(newA, data_add, 0)
            newA = newA[newA[:, -1]. argsort()]  #сортировка по изначальным номерам строк
            newA = np.delete(newA, -1, 1)
            mtrxA = newA
        #print("mtrxA", mtrxA)    
        
        #Back substitution
        for row_i in range(size - 1, -1, -1): 
            vecX[row_i] = mtrxA[row_i][size]
            for row_j in range(size - 1, row_i, -1):
                vecX[row_i] -= mtrxA[row_i][row_j] * vecX[row_j]
            vecX[row_i] = vecX[row_i] / mtrxA[row_i][row_i]
        #print(vecX)
        
    if (rank > 0):
        for i in range(size - 1):
            data = comm.recv(source=0, tag = i)
            if (len(data) > 1):
                new_data = elimination(data, rank)
                comm.send(new_data, dest=0, tag=rank) 
               
    if (rank == 0):
        np.savetxt('x_mpi_py.txt', vecX, fmt='%10.5f')
        end_time = time.time()
        print('Time:{}s'.format(end_time - start_time))
        print('MPI/Python: size of matrice', size, ', processes', numprocs)
        


