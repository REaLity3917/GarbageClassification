import numpy as np
import cv2

# 通过九点标定获取的圆心相机坐标
STC_points_camera = np.array([
    [192, 131],
    [313, 129],
    [434, 132],
    [188, 252],
    [311, 252],
    [435, 253],
    [190, 371],
    [311, 375],
    [432, 374]
])
# 通过九点标定获取的圆心机械臂坐标
STC_points_robot = np.array([
    [160, 350],
    [205, 480],
    [410, 620],
    [205, 230],
    [410, 350],
    [550, 480],
    [410, 80],
    [580, 200],
    [760, 350]
])


# 手眼标定方法
class HandInEyeCalibration:

    def get_m(self, points_camera, points_robot):
        """
        取得相机坐标转换到机器坐标的仿射矩阵
        :param points_camera:
        :param points_robot:
        :return:
        """
        # 确保两个点集的数量级不要差距过大，否则会输出None
        m, _ = cv2.estimateAffine2D(points_camera, points_robot)
        return m

    def get_points_robot(self,m,x_camera, y_camera):
        """
        相机坐标通过仿射矩阵变换取得机器坐标
        :param x_camera:
        :param y_camera:
        :return:
        """

        robot_x = (m[0][0] * x_camera) + (m[0][1] * y_camera) + m[0][2]
        robot_y = (m[1][0] * x_camera) + (m[1][1] * y_camera) + m[1][2]
        return robot_x, robot_y

if __name__ == "__main__":
    # 声明类
    H=HandInEyeCalibration()
    # 获取转换矩阵
    m = H.get_m(STC_points_camera, STC_points_robot)
    #根据像素坐标计算机械坐标

    robot_x, robot_y=H.get_points_robot(m,207,160)
    print(robot_x,robot_y)