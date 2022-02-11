import numpy as np
import argparse
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument("input_n",type=int)
args=parser.parse_args()
n=args.input_n
A=np.zeros((n,n))
b=np.zeros((n,1))
C=np.zeros((n,1))
random.seed(time.time())
t0_start=time.perf_counter()
for i in range(n):
    for j in range(n):
        A[i,j]=random.uniform(-1,1)
for i in range(b.shape[0]):
    b[i]=1.0
t0_end=time.perf_counter()
#print((t0_end-t0_start)/20)



for step in range(20):
    t1_start=time.perf_counter()
    for i in range(n):
        new_sum=0
        for j in range(n):
            new_sum=new_sum+A[i,j]*b[j]
        C[i]=new_sum
    t1_end=time.perf_counter()
    t1=float((t1_end-t1_start)*1000)
    print("1 {} {:.3f}".format(n,t1))


for step in range(20):
    t2_start=time.perf_counter()
    C=np.matmul(A,b)
    t2_end=time.perf_counter()
    t2=(t2_end-t2_start)*1000
    print("2 {} {:.3f}".format(n,t2))
