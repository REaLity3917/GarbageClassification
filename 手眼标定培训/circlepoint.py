import cv2

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 参数0表示使用默认摄像头

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

# 创建窗口
cv2.namedWindow("camera")

# 用于存储所有点击位置的坐标
click_positions = []

# 定义鼠标回调函数
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 将点击位置的坐标添加到列表中
        click_positions.append((x, y))
        print(f"鼠标点击位置的像素坐标: ({x}, {y})")

# 设置鼠标回调函数
cv2.setMouseCallback("camera", on_EVENT_LBUTTONDOWN)

while True:
    # 读取摄像头的每一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧！")
        break

    # 调整帧的大小
    frame = cv2.resize(frame, (800, 600))  # 将帧调整为800x600

    # 裁剪帧以显示中心区域
    frame = frame[(600 // 2 - 240):(600 // 2 + 240), (800 // 2 - 240):(800 // 2 + 240)]

    # 在每一帧上绘制所有点击位置的圆点和坐标
    for pos in click_positions:
        x, y = pos
        # 检查点击位置是否在裁剪后的区域内
        if 0 <= x < 480 and 0 <= y < 480:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # 红色圆点
            cv2.putText(frame, f"{x},{y}", (x + 10, y + 10), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)  # 黑色文本

    # 显示帧
    cv2.imshow("camera", frame)

    # 按下'Esc'键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()