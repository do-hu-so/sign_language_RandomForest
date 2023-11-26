import cv2
import os
import mediapipe as mp
import pickle
#sử dụng mediapipe để nhận diện landmarks bàn tay

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.4)

#đường dẫn lưu trữ hình ảnh
DATA_DIR  = r'D:\Computer_vision_basic\pass5\hand_gesture\data'

data=[] #danh sách chứa các vector đặc chưng cyar landmarks
labels =[] #danh sách chứa các nhãn của mỗi vector đặc trưng


#lặp qua từng thu mục class của thư mục data
for dir in os.listdir(DATA_DIR):
    #lặp qua từng ảnh trong từng thư mục class
    for img_path in os.listdir(os.path.join(DATA_DIR,dir)):
        data_aux =[] #danh sách chứa vector đặc chưng của mỗi hình ảnh : x, y của 21 điểm\
        x_ =[] #danh sách chứa tọa độ x của landmarks
        y_=[] #danh sách chứa tọa độ y của landmarks

        img = cv2.imread(os.path.join(DATA_DIR,dir,img_path)) # data/0/0.jpg ....
        img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #nhận diện landmarks
        results = hands.process(img_rbg)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                #lặp qua 21 point 
                for i in range(len(hand_landmarks.landmark)):
                    x= hand_landmarks.landmark[i].x
                    y= hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                #chuẩn hóa tọa độ của các landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x-min(x_))
                    data_aux.append(y-min(y_))
            #thêm vector đặc trưng và nhãn tương ứng vào danh sách
            data.append(data_aux)
            labels.append(dir)
#lưu dữ liệu và nhãn vào file pickle
with open("data.pickle", 'wb') as f:
    pickle.dump({'data': data, 'labels': labels},f)

        