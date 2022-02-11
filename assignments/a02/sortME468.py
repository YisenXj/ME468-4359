import numpy as np
import argparse
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument("input_n",type=int)
args=parser.parse_args()
n=args.input_n
data=np.zeros(n)

random.seed(time.time())

for i in range(n):
    data[i]=random.randrange(0,1000)

print(int(data[0]),int(data[n-1]))
for i in range(n-1):
    for j in range(i+1,n):
        if data[i]>data[j]:
            t=data[i]
            data[i]=data[j]
            data[j]=t
print(int(data[0]),int(data[n-1]))


