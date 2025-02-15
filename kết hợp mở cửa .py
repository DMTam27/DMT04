import pandas as pd
import cv2
import requests
import numpy as np
import os
import serial
from datetime import datetime
import face_recognition

  # Đảm bảo cổng đúng với cổng Arduino của bạn

# Đường dẫn đến thư mục chứa các hình ảnh
path = r'C:\Users\dell\Downloads\ATTENDANCE\image_folder'
url = 'http://192.168.91.149/cam-hi.jpg'

# Kiểm tra và tạo file Attendance.csv nếu chưa tồn tại
attendance_path = os.path.join(os.getcwd(), 'attendance')
if not os.path.exists(attendance_path):
    os.makedirs(attendance_path)

attendance_file = os.path.join(attendance_path, 'Attendance.csv')
if os.path.exists(attendance_file):
    os.remove(attendance_file)
else:
    df = pd.DataFrame(list())
    df.to_csv(attendance_file)

images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open(attendance_file, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

def put_text(img, text, pos, font_scale, font_thickness, color=(0, 255, 0), max_width=200):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    while text_size[0] > max_width:
        font_scale -= 0.1
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    text_pos = (pos[0], pos[1] + text_size[1])
    cv2.putText(img, text, text_pos, font, font_scale, color, font_thickness)

while True:
    try:
        response = requests.get(url, timeout=20)  # Sử dụng requests và thêm thời gian chờ lớn hơn
        response.raise_for_status()  # Kiểm tra xem có lỗi HTTP nào không
        imgnp = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)
    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')
        continue

    # Giảm kích thước và chuyển đổi màu sắc
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Kiểm tra nếu không có khuôn mặt nào đủ gần
        if min(faceDis) > 0.6:  # Ngưỡng khoảng cách có thể thay đổi tùy theo yêu cầu
            name = "Unknown"  # Đặt là "Unknown" nếu không nhận diện được
        else:
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
        put_text(img, name, (x1 + 6, y2 - 6), font_scale=1, font_thickness=2, color=(255, 255, 255), max_width=(x2-x1))
        if name != "Unknown":
            markAttendance(name)  # Chỉ lưu khi nhận diện được khuôn mặt đã đăng ký
            
            # Gửi dữ liệu đến Arduino để mở cửa
            arduino.write(b'UNLOCK_DOOR\n')  # Gửi lệnh mở cửa qua Serial

            # Hiển thị khung màu xanh (mở cửa)
            cv2.imshow('Webcam', img)
            cv2.waitKey(500)  # Hiển thị khung màu lâu hơn (0.5 giây)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
