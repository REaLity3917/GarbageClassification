import cv2
import os

# 获取当前工作目录
dir = os.getcwd()
type = 'mixaaaaaa'
count = 0
if not os.path.exists(type):
    os.mkdir(type)
path = os.path.join(dir, type)
cap = cv2.VideoCapture(0)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = 900
frame_height = 675

# 计算中心区域的坐标
start_x = (frame_width - 640) // 2
start_y = (frame_height - 480) // 2
end_x = start_x + 640
end_y = start_y + 480

# 检查已有文件并更新计数器
existing_files = [f for f in os.listdir(path) if f.startswith(type)]
if existing_files:
    existing_numbers = [int(f[len(type):-4]) for f in existing_files]
    count = max(existing_numbers) + 1

def take_photo(frame):
    global path, count, type, cap

    # 生成图片名称
    name = type + str(count) + '.jpg'
    file_path = os.path.join(path, name)
    cv2.imwrite(file_path, frame)
    print("照片已保存至", file_path)
    count += 1

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (900, 675))
    cv2.rectangle(frame, (start_x-1, start_y-1), (end_x+1, end_y+1), (0, 255, 0), 1)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    # 按下q键，拍照并退出循环
    if key == ord('q'):
        cropped_frame = frame[start_y:end_y, start_x:end_x]
        take_photo(cropped_frame)

    # 按下w键，退出循环
    if key == ord('w'):
        break

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
