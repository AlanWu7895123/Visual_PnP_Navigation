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

#include "abstract_state_machine.h"

using namespace pcl;
using namespace cv;
using namespace std;
using namespace efsm;

class AGVSystem_start_param : public start_param_base
{
public:
    using ptr = std::shared_ptr<AGVSystem_start_param>;

    double x;
    double y;
};

class AGVSystem final : public abstract_state_machine
{
public:
    using ptr = std::shared_ptr<AGVSystem>;
    AGVSystem(int type); // 构造函数，初始化AGV系统

    bool restart(start_param_base::ptr) override;
    bool pause() override;
    bool resume() override;
    bool stop() override;

    void run_once() override;

    void run(); // 启动AGV系统的主要功能
    void setTarget(double x, double y);
    State getCurrentState();

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

    AGVSystem_start_param::ptr start_param_{nullptr};

    // 添加其他必要的成员变量
};

#endif // AGV_SYSTEM_H
