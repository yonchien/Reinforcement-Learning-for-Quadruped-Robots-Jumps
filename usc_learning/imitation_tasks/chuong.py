import numpy as np

a=np.array([1, 2, 3]*4)
a1=np.array([1, 2, 3])
a2=np.array([[1, 2, 3]]*4)
a3=[a1]*4
# a=a.transpose()
print(a)
print(np.array(a1*4))
print(a2)
print("a3:",a3)
print(np.tile(a1,4))
# b=np.repeat(a, 3)
# print(b)
# c=np.concatenate(a, 1)
# print(c)

# d=[np.zeros(2)] * 5
# print(d)
# print(len(d))