import json
import numpy as np
import torch
from torch_geometric.data import Data


with open('edinburgh-keywords_train.json', 'r') as f:
    train_data = json.load(f)

keywords = list(train_data['np2count'].keys())

# Loại bỏ những từ bị trùng
keyword_set = set(keywords)

# Danh sách các nhà hàng trong tập train
restaurants = list(train_data['npr2rest'].keys())
restaurant_set = set(restaurants)

# Tạo ma trận liên kết từ keywords và restaurant
num_keywords = len(keyword_set)
num_restaurants = len(restaurants)
a = np.zeros((num_keywords, num_restaurants))


for i, keyword in enumerate(keyword_set):
    for j, restaurant in enumerate(restaurants):
        if keyword in train_data['npr2rest'][restaurant]['reviews']:
            a[i, j] = 1


# Ma trận liên kết giữa keywords và restaurants
print(a)
with open('edinburgh-ketywords_test.json', 'r') as r:
    test_data = json.load(r)

user_keywords = list(test_data['np2reviews'].keys())

user_keywords_set = set(user_keywords)

# Ma trận keywords của users
t = np.array(user_keywords)

# Nhân ma trận 
R = np.dot(t, a)

# Kết quả là mức độ liên kết của người dùng với các nhà hàng
print("Mức độ liên kết giữa người dùng và nhà hàng:", R)

# Dữ liệu: từ khóa (keywords) và nhà hàng (restaurants) là các node
# Giả sử có 5 từ khóa và 3 nhà hàng, với ma trận liên kết a[i][j]
edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 1, 2]], dtype=torch.long)  # Connection between nodes

x = torch.tensor([[1], [1], [1], [1], [1]], dtype=torch.float)  # Node là các keywords hoặc restaurants

data = Data(x=x, edge_index=edge_index)