import tensorflow as tf
import numpy as np
import pdb
from math import sqrt, cos, sin, pi

S = 1
K = 2
N = 3
zdim = 5
C = 1.

# norm_shpe = [-1,K]
# sample_x = tf.constant(np.arange(S*K*zdim,dtype=np.float32).reshape((S,K,zdim)))
# norms_x = tf.reduce_sum(tf.square(sample_x), axis=-1)
# nx_reshpe = tf.reshape(norms_x,norm_shpe+[1,1])
#
# sample_y = tf.constant(np.arange(0,2*S*K*zdim,2,dtype=np.float32).reshape((S,K,zdim)))
# norms_y = tf.reduce_sum(tf.square(sample_y), axis=-1)
# ny_reshpe = tf.reshape(norms_x,[1,1]+norm_shpe)
#
# dotprod = tf.tensordot(sample_x, sample_y, [[-1],[-1]])
# distances = nx_reshpe + ny_reshpe - 2. * dotprod

distances = tf.constant(np.arange(N*K*S*N*K*S,dtype=np.float32).reshape((N,K,S,N,K,S)))
distances += tf.transpose(distances,perm=[0,1,5,3,4,2])
distances /= 2.

# qz term
# K_qz = tf.reduce_sum(distances,axis=[2,-1])
# res1_qz = tf.reduce_sum(K_qz)
# res2_qz = tf.trace(tf.reduce_sum(K_qz,axis=[1,3]))
# res_qz = res1_qz - res2_qz


K_pz = tf.reduce_sum(distances,axis=[2,-1])
res1_pz = tf.reduce_sum(K_pz)
res1_pz /= (N * N)
res2_pz = tf.trace(tf.reduce_sum(K_pz,axis=[0,2]))
#res2_pz *= (1. / (N * N - N) - 1. / (N * N))
res2_pz /= ((N * N - N) * N)
res3_pz = tf.trace(tf.trace(tf.transpose(K_pz,perm=[0,2,1,3])))
res3_pz /= (N * N - N)
res_pz = res1_pz + res2_pz - res3_pz


def sqre_dist(x,y):
    return np.sum(np.square(x-y))

sess = tf.Session()
# x = sess.run(sample_x)
# y = sess.run(sample_y)
# xy = sess.run(dotprod)
# nx = sess.run(norms_x)
# nx_re = sess.run(norms_x)
# ny = sess.run(norms_y)
# ny_re = sess.run(norms_y)
# xy = sess.run(dotprod)
images = sess.run(tf.image.grayscale_to_rgb(np.ones((10,28,28,1))))
pdb.set_trace()
di = sess.run(distances)
K_pz = sess.run(K_pz)
# res1_qz = sess.run(res1_qz)
# res2_qz = sess.run(res2_qz)
res_pz = sess.run(res_pz)

res1 = 0.
res2 = 0.
for k in range(K):
    for p in range(K):
        for i in range(N):
            for j in range(N):
                if p!=k:
                    for m in range(S):
                        for l in range(S):
                            res1 += di[i,k,m,j,p,l]
                else:
                    if j!=i:
                        for m in range(S):
                            for l in range(S):
                                res2 += di[i,k,m,j,p,l]
res = res1 / (N * N) + res2 / (N * N - N)
print( res==res_pz)
# log = sess.run(logits)
# lmax = sess.run(l_max)
# lse_shift = sess.run(logsumexp_shifted)
# lse = sess.run(logsumexp)

# input = sess.run(input)
# input_reshape = sess.run(input_reshape)
# l1 = sess.run(l1)
# output = sess.run(output)
# zero = sess.run(zero)
# one = sess.run(one)
# two = sess.run(two)
# stack = sess.run(stack)
# reshape = sess.run(reshape)
# y1 = sess.run(y1)
# y2 = sess.run(y2)
# sqr_dif = sess.run(sqr_dif)
# c = sess.run(c)
# idx_label = sess.run(idx_label)
# l_pi = sess.run(l_pi)
# x = sess.run(sample_x)
# nx = sess.run(norms_x)
# xx = sess.run(dotprod_x)
# dx = sess.run(distances_x)
# y = sess.run(sample_y)
# ny = sess.run(norms_y)
# yy = sess.run(dotprod_y)
# dy = sess.run(distances_y)
# dxy = sess.run(distances)
# diag = sess.run(diag)
# out = sess.run(out)
# rp = sess.run(res1_pz)
# rq = sess.run(res1_qz)
# r1 = sess.run(res1)
# r1_diag = sess.run(res1_diag)
# r1_corr = sess.run(res1_corr)
# r2 = sess.run(res2)
# r = sess.run(res)

pdb.set_trace()
