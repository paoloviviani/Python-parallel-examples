import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Array
from itertools import product
import math
import dask.array as da
import ray
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--size', type=int, default=1000,
                    help='Generate size x size image')
parser.add_argument('--nproc', type=int, default=4,
                    help='number of processes to spawn')

args = parser.parse_args()

n = args.size
nproc = args.nproc
ray.init(num_cpus=nproc)

def get_iter(c:complex, thresh:int =4, max_steps:int =25) -> int:
    # Z_(n) = (Z_(n-1))^2 + c
    # Z_(0) = c
    z=c
    i=1
    while i<max_steps and (z*z.conjugate()).real<thresh:
        z=z*z +c
        i+=1
    return i

@ray.remote
def prociter(idx, thresh, max_steps=25):
    print(idx)
    img=np.zeros((n,n))
    mx = 2.48 / (n-1)
    my = 2.26 / (n-1)
    mapper = lambda x,y: (mx*x - 2, my*y - 1.13)
    for i in idx:
        x = math.floor(i/n)
        y = i%n
        it = get_iter(complex(*mapper(x,y)), thresh=thresh, max_steps=max_steps)
        img[y,x] = it
    return img

def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))  

def plotter(n, thresh, max_steps=25):
    idx=split(range(n*n),nproc)
    results = ray.get( [prociter.remote(item, thresh, max_steps) for item in idx] )
    out = np.full((n,n), 255)
    for item in results:
        out = out - item

    return out

start = timer()
img = plotter(n, thresh=4, max_steps=100)
end = timer()
print('TIME: '+ str(end-start))

plt.imshow(img, cmap="plasma")
plt.axis("off")
plt.savefig('mandelbrot.png', format="png")