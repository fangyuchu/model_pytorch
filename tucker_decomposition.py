import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac   #CPdecomposition

X = tl.tensor(np.arange(24,dtype=np.float32).reshape(3, 4, 2))
# 使用parafac将张量进行CP分解
factors = parafac(X, rank=3)

print(factors)    #打印出三个维度矩阵 ，列为rank，行为张量的行数
print(len(factors))
print([f.shape for f in factors])

# 分解为三个维度矩阵，矩阵的行对应张量的每一个维度，列对应rank
# 将分解之后的矩阵还原成原始张量，并进行比较
full_tensor = tl.kruskal_to_tensor(factors)
print("original_tensor: % r" % X)
print("full_tensor: % r" % full_tensor)

# def decompose_three_way(tensor, rank, max_iter=501, verbose=False):
#  # a = np.random.random((rank, tensor.shape[0]))
#  b = np.random.random((rank, tensor.shape[1]))
#  c = np.random.random((rank, tensor.shape[2]))
#  for epoch in range(max_iter):
#      # optimize a
#      input_a = khatri_rao([b.T, c.T])
#      target_a = tl.unfold(tensor, mode=0).T
#      a = np.linalg.solve(input_a.T.dot(input_a), input_a.T.dot(target_a))
#      # optimize b
#      input_b = khatri_rao([a.T, c.T])
#      target_b = tl.unfold(tensor, mode=1).T
#      b = np.linalg.solve(input_b.T.dot(input_b), input_b.T.dot(target_b))
#      # optimize c
#      input_c = khatri_rao([a.T, b.T])
#      target_c = tl.unfold(tensor, mode=2).T
#      c = np.linalg.solve(input_c.T.dot(input_c), input_c.T.dot(target_c))
#      if verbose and epoch % int(max_iter * .2) == 0:
#          res_a = np.square(input_a.dot(a) - target_a)
#          res_b = np.square(input_b.dot(b) - target_b)
#          res_c = np.square(input_c.dot(c) - target_c)
#          print("Epoch:", epoch, "| Loss (C):", res_a.mean(), "| Loss (B):", res_b.mean(), "| Loss (C):", res_c.mean())
#  return a.T, b.T, c.T

