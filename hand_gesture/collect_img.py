import cv2
import os

#tạo thư mục để lưu dữ liệu 
DATA_DIR  = r'D:\Computer_vision_basic\pass5\hand_gesture\data'
#print(os.path.exists(DATA_DIR)) # dùng để kiểm tra xem có đường dẫn hay không

classes = 6 #['mot','hai',....]
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(classes):
    #tạo 1 folder mới tương ứng với mỗi class
    if not os.path.exists(os.path.join(DATA_DIR,str(j))):
        os.makedirs(os.path.join(DATA_DIR,str(j)))
    stemp=0
    #quá trình chuẩn bị tư thế tay trước khi thu thập dữ liệu
    while True:
        secc, frame = cap.read()
        cv2.putText(frame, f" chuan bi thu class{j}, anh phim 'Q': ",(100,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.3, (0,255,0),3,
                    cv2.LINE_AA)
        cv2.imshow('img',frame)   
        if cv2.waitKey(25) == ord('q'):
            break

    #quá trìn thu thập dữ liệu
    while stemp<dataset_size:
        secc, frame = cap.read()
        cv2.imshow('img',frame)
        cv2. imwrite(os.path.join(DATA_DIR,str(j), f'{stemp}.jpg'), frame) # daa/0/0.jpg 
        stemp+=1
        cv2.waitKey(25)


#giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()