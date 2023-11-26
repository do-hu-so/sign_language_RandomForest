# Import các thư viện cần thiết
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load mô hình đã được huấn luyện từ file 'model.p'
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Khởi tạo VideoCapture để đọc dữ liệu từ camera (0 là camera mặc định)
cap = cv2.VideoCapture(0)

# Khởi tạo đối tượng mediapipe cho việc xử lý tay
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Vòng lặp chính của chương trình
while True:
    # Khởi tạo danh sách và biến cho việc lưu trữ dữ liệu tọa độ tay
    data_aux = []
    x_ = []
    y_ = []
    # Đọc frame từ camera
    ret, frame = cap.read()

    # Lấy chiều cao và chiều rộng của frame
    H, W, _ = frame.shape
    
    # Chuyển đổi frame sang không gian màu RGB để sử dụng với mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý dữ liệu tay bằng mediapipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Vẽ các đường kết nối và đặc trưng của tay lên frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # ảnh để vẽ
                hand_landmarks,  # kết quả từ mediapipe
                mp_hands.HAND_CONNECTIONS,  # kết nối các điểm trên tay
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Thu thập dữ liệu tọa độ tay
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Dự đoán nhãn tương ứng với dữ liệu tọa độ tay
        prediction = model.predict([np.asarray(data_aux)]) #chuyển đổi một đối tượng có thể chuyển đổi thành mảng NumPy
        print(prediction[0])

        # Tính toán tọa độ của hình chữ nhật bao quanh tay
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        # Vẽ hình chữ nhật và nhãn dự đoán lên frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, prediction[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Hiển thị frame
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# Giải phóng tài nguyên và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
