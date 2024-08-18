#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "utils.h"
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

class Controller
{
public:
    Controller(const std::string &ipAddress, const std::string &port);
    ~Controller();
    bool connectToServer();
    bool sendData(const ProtocolHeader &header, const std::string &jsonData);
    bool receiveData(ProtocolHeader &header, std::string &jsonData);
    void cleanup();

private:
    std::string ipAddress;
    std::string port;
    int ConnectSocket;
};
#endif // CONTROLLER_H
