import pickle
path =r"D:\Computer_vision_basic\pass5\data.pickle"
data = pickle.load(open(path,'rb'))
a = data.keys()
print(a)
print(data)