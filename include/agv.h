#ifndef AGV_H
#define AGV_H

#include "controller.h"
#include "struct.h"
#include "json.hpp"

class AGV
{
public:
    AGV(string ip);
    nlohmann::json sendToAGV(Controller &controller, uint16_t type, std::string jsonData);
    nlohmann::json move(double dist, double vx, double vy);
    nlohmann::json rotate(double angle, double vw);
    nlohmann::json getState();
    nlohmann::json setPermission();

private:
    Controller configController; // 配置API
    Controller stateController;  // 状态API
    Controller moveController;   // 控制API
};

#endif // AGV_H
