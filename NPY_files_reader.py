import numpy as np

y = np.load(r"/Users/kevin/PycharmProjects/TensorCat/CV/Sign_Language_Model/NPY Dataset/y_TRAIN_2.npy")
print(y.shape, y.dtype)    # (3387,) int64
print(y[:1000])              # 看前20个标签
print(y.min(), y.max())    # 看标签范围