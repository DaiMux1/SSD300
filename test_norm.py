import numpy as np
 
X = np.array([4, 3])

#Norm: chuaan hóa để tính ra 1 đại lượng biểu diễn độ lơn của 1 vector 

#L0Norm: so pha tu khac 0
#l0Norm=2

l0norm = np.linalg.norm(X, ord=0)
print(l0norm)

#L1Norm: khoảng cách mahattan
l1norm = np.linalg.norm(X, ord=1)
print(l1norm)

l2 = np.linalg.norm(X, ord=2)
print(l2)

