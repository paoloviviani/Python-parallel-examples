import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Array
from itertools import product
import math
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--size', type=int, default=1000,
                    help='Generate size x size image')
parser.add_argument('--nproc', type=int, default=4,
                    help='number of processes to spawn')
parser.add_argument('--fake_work', action='store_true')
parser.add_argument('--no_lock', action='store_false')

args = parser.parse_args()

n = args.size
nproc = args.nproc
glock = args.no_lock
artificial_work = args.fake_work # Artificially double the work and overlap output forcing lock on shared array to kick in

def get_iter(c:complex, thresh:int =4, max_steps:int =25) -> int:
    # Z_(n) = (Z_(n-1))^2 + c
    # Z_(0) = c
    z=c
    i=1
    while i<max_steps and (z*z.conjugate()).real<thresh:
        z=z*z +c
        i+=1
    return i

def prociter(idx, outarr, thresh, max_steps=25):
    mx = 2.48 / (n-1)
    my = 2.26 / (n-1)
    mapper = lambda x,y: (mx*x - 2, my*y - 1.13)
    for i in idx:
        x = math.floor(i/n)
        y = i%n
        it = get_iter(complex(*mapper(x,y)), thresh=thresh, max_steps=max_steps)
        outarr[n*y+x] = 255 - it

def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))  

def plotter(n, thresh, max_steps=25):
    imgArr = Array('d', [255]*n*n, lock=glock)
    idx=split(range(n*n),nproc)
    procs = []
    for item in idx:
        proc = mp.Process(target=prociter, args=(item, imgArr, thresh, max_steps))
        procs.append(proc)
        proc.start()
        if artificial_work:
            proc1 = mp.Process(target=prociter, args=(item, imgArr, thresh, max_steps))
            procs.append(proc1)
            proc1.start()
    
    for proc in procs:
        proc.join()

    return imgArr

start = timer()
imgArr = plotter(n, thresh=4, max_steps=100)
end = timer()
print('TIME: '+ str(end-start))

if glock:
    img = np.frombuffer(imgArr.get_obj()).reshape((n, n))
else:
    img = np.array(imgArr).reshape((n, n))

plt.imshow(img, cmap="plasma")
plt.axis("off")
plt.savefig('mandelbrot.png', format="png")