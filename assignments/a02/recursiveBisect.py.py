import numpy as np
import math


def foo(x):
    return math.exp(x)-4


x_low=-1
x_high=2
step=0;
while x_high-x_low>1e-6:
    step=step+1
    x_half=x_low/2+x_high/2
    print("Interation %d: %f %f %f %f" % (step,x_low,x_high,x_half,foo(x_half)))
    if foo(x_half)*foo(x_low)>0:
        x_low=x_half
    else:
        x_high=x_half
