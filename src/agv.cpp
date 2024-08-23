#include "agv.h"

AGV::AGV(string ip) : configController(ip, "19207"), stateController(ip, "19204"), moveController(ip, "19206") {}

nlohmann::json AGV::sendToAGV(Controller &controller, uint16_t type, std::string jsonData)
{
    if (!controller.connectToServer())
    {
        std::cerr << "Failed to connect to server." << std::endl;
        return nullptr;
    }

    std::cout << "Send json data = " << jsonData << std::endl;

    ProtocolHeader header;
    header.m_sync = 0x5A;
    header.m_version = 0x01;
    // header.m_number = htons(1); // Network byte order
    header.m_number = 1;
    memset(header.m_reserved, 0, sizeof(header.m_reserved));

    // header.m_type = htons(type);              // Network byte order
    // header.m_length = htonl(jsonData.size()); // Network byte order
    header.m_type = type;
    header.m_length = jsonData.size();

    if (!controller.sendData(header, jsonData))
    {
        std::cerr << "Failed to send data." << std::endl;
        return nullptr;
    }

    std::cout << "Ready to receive response" << std::endl;

    ProtocolHeader responseHeader;
    std::string responseJsonData;
    if (!controller.receiveData(responseHeader, responseJsonData))
    {
        std::cerr << "Failed to receive data." << std::endl;
        return nullptr;
    }

    controller.cleanup();
    return nlohmann::json::parse(responseJsonData);
}

nlohmann::json AGV::move(double dist, double vx, double vy)
{
    AGVTranslation translation;
    translation.dist = dist;
    translation.vx = vx;
    translation.vy = vy;

    nlohmann::json jTranslation;
    jTranslation["dist"] = translation.dist;
    jTranslation["vx"] = translation.vx;
    jTranslation["vy"] = translation.vy;
    std::string jStrTranslation = jTranslation.dump();

    nlohmann::json translationResponseJson = sendToAGV(moveController, 3055, jStrTranslation);

    std::cout << "waiting\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    return translationResponseJson;
}

nlohmann::json AGV::rotate(double angle, double vw)
{
    AGVRotation rotation;
    rotation.angle = angle;
    rotation.vw = vw;

    nlohmann::json jRotation;
    jRotation["angle"] = rotation.angle;
    jRotation["vw"] = rotation.vw;
    std::string jStrRotation = jRotation.dump();

    nlohmann::json rotationResponseJson = sendToAGV(moveController, 3056, jStrRotation);

    std::cout << "waiting\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    return rotationResponseJson;
}

nlohmann::json AGV::getState()
{
    string jsonData = "";
    nlohmann::json responseJson = sendToAGV(stateController, 1000, jsonData);
    return responseJson;
}

nlohmann::json AGV::getSpeed()
{
    string jsonData = "";
    nlohmann::json responseJson = sendToAGV(stateController, 1005, jsonData);
    return responseJson;
}

nlohmann::json AGV::getIfBlocked()
{
    string jsonData = "";
    nlohmann::json responseJson = sendToAGV(stateController, 1006, jsonData);
    return responseJson;
}

nlohmann::json AGV::getIMU()
{
    string jsonData = "";
    nlohmann::json responseJson = sendToAGV(stateController, 1014, jsonData);
    return responseJson;
}

nlohmann::json AGV::setPermission()
{
    nlohmann::json jConfig;
    jConfig["nick_name"] = "srd-seer-mizhan";
    std::string jStrConfig = jConfig.dump();
    nlohmann::json configResponseJson = sendToAGV(configController, 4005, jStrConfig);
    std::cout << "waiting\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    return configResponseJson;
}