#pragma once

enum class State
{
    INIT,
    INIT_RUNNING,
    TEST,
    TEST_RUNNING,
    CAPUTRE_IMAGE,
    CAPUTRE_IMAGE_RUNNING,
    FEATURE_DETECT,
    FEATURE_DETECT_RUNNING,
    MATCHING,
    MATCHING_RUNNING,
    MAPPING,
    MAPPING_RUNNING,
    MOVING,
    MOVING_RUNNING,
    FINISHED
};

enum class AGVMoveStep
{
    WAITINGFORTARGET,
    ROTATE,
    FORWARD,
    BACK
};

struct AGVTranslation
{
    double dist; // 直线运动距离, 绝对值, 单位: m
    double vx;   // 机器人坐标系下 X 方向运动的速度, 正为向前, 负为向后, 单位: m/s
    double vy;   // 机器人坐标系下 Y 方向运动的速度, 正为向左, 负为向右, 单位: m/s
};

struct AGVRotation
{
    double angle; // 转动的角度(机器人坐标系), 绝对值, 单位 rad, 可以大于 2π
    double vw;    // 转动的角速度(机器人坐标系), 正为逆时针转, 负为顺时针转 单位 rad/s
};

struct ProtocolHeader
{
    uint8_t m_sync;
    uint8_t m_version;
    uint16_t m_number;
    uint32_t m_length;
    uint16_t m_type;
    uint8_t m_reserved[6];
};