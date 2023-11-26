import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import numpy as np

#load dữ liệu từ file pickle
path=r'D:\Computer_vision_basic\pass5\data.pickle'

data_dict = pickle.load(open(path,'rb'))

#chuyển đổi dữ liệu và nhãn thành mảng numpy
data= np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


# chọn ra các đặc chưng cần sử dụng: cứ 42 đặc trưng 1 lần
X = data[:,:42]

#chia tập dữ liệu thành train  và test :
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, shuffle=True, stratify=labels)

#khởi động và huấn luyện mô hình RandomForest

model = RandomForestClassifier()
model.fit(x_train,y_train)

#tự dự đoán trên tập test
y_predict = model.predict(x_test)
#đánh giá mô hình
score = accuracy_score(y_predict,y_test)
#in ra kết quả sau trainning

print(f'{score*100}% của mô hình đã training')


with open('model.p', 'wb') as f:
    pickle.dump({'model': model},f)