#ifndef AGVSYSTEM_H
#define AGV_SYSTEM_H

#include <mutex>
#include <condition_variable>
#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include "camera.h"
#include "filter.h"
#include "detect.h"
#include "icp_algorithm.h"
#include "pnp.h"
#include "utils.h"
#include "network_camera.h"
#include "json.hpp"
#include "controller.h"
#include "agv.h"
#include "constant.h"

class AGVSystem
{
public:
    AGVSystem(int type); // 构造函数，初始化AGV系统
    void run();          // 启动AGV系统的主要功能
    void setTarget(double x, double y);

private:
    // 初始化线程
    void initThread();

    // 结束线程
    void finishThread();

    // 图像捕获线程
    void captureImageThread();

    // 特征检测线程
    void featureDetectThread();

    // 点集配准线程
    void matchingThread();

    // 地图映射线程
    void mappingThread();

    // AGV移动控制线程
    void movingThread();

    // 测试线程（可选）
    void testThread();

    std::ofstream _angle_out, _agvPose_out, _t_out; // 输出文件流

    // 添加其他必要的成员变量
};

#endif // AGV_SYSTEM_H
