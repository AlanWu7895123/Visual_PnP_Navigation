#include "controller.h"
#include <iostream>
#include <cstring> // for memset
#include <errno.h> // for errno

Controller::Controller(const std::string &ipAddress, const std::string &port)
    : ipAddress(ipAddress), port(port), ConnectSocket(-1)
{
}

Controller::~Controller()
{
    cleanup();
}

void Controller::cleanup()
{
    if (ConnectSocket != -1)
    {
        close(ConnectSocket);
    }
}

bool Controller::connectToServer()
{
    struct addrinfo hints, *result, *ptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    int iResult = getaddrinfo(ipAddress.c_str(), port.c_str(), &hints, &result);
    if (iResult != 0)
    {
        std::cerr << "getaddrinfo failed with error: " << gai_strerror(iResult) << std::endl;
        return false;
    }

    for (ptr = result; ptr != NULL; ptr = ptr->ai_next)
    {
        ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
        if (ConnectSocket == -1)
        {
            std::cerr << "socket failed with error: " << strerror(errno) << std::endl;
            continue;
        }

        iResult = connect(ConnectSocket, ptr->ai_addr, ptr->ai_addrlen);
        if (iResult == -1)
        {
            close(ConnectSocket);
            ConnectSocket = -1;
            continue;
        }
        break;
    }

    freeaddrinfo(result);

    if (ConnectSocket == -1)
    {
        std::cerr << "Unable to connect to server!" << std::endl;
        return false;
    }

    return true;
}

bool Controller::sendData(const ProtocolHeader &header, const std::string &jsonData)
{
    std::string sendData = serializeProtocolHeader(header) + jsonData;
    const char *sendBuf = sendData.c_str();

    ssize_t iResult = send(ConnectSocket, sendBuf, sendData.size(), 0);
    if (iResult == -1)
    {
        std::cerr << "send failed with error: " << strerror(errno) << std::endl;
        return false;
    }

    std::cout << "Bytes Sent: " << iResult << std::endl;
    std::cout << "Content Sent: " << bytesToHex(sendBuf, sendData.size()) << std::endl;

    return true;
}

bool Controller::receiveData(ProtocolHeader &header, std::string &jsonData)
{
    char *headerBuf = new char[sizeof(ProtocolHeader)];
    int totalBytesReceived = 0;
    int bytesReceived = 0;

    while (totalBytesReceived < sizeof(ProtocolHeader))
    {
        bytesReceived = recv(ConnectSocket, headerBuf + totalBytesReceived, sizeof(ProtocolHeader) - totalBytesReceived, 0);
        if (bytesReceived < 0)
        {
            std::cerr << "Error receiving header. Error: " << strerror(errno) << std::endl;
            return false;
        }
        totalBytesReceived += bytesReceived;
    }

    header = deSerializeProtocolHeader(headerBuf);
    std::cout << "m_length=" << header.m_length << std::endl;
    std::cout << "m_type=" << header.m_type << std::endl;

    char *jsonDataBuf = new char[header.m_length + 1];
    totalBytesReceived = 0;

    while (totalBytesReceived < header.m_length)
    {
        bytesReceived = recv(ConnectSocket, jsonDataBuf + totalBytesReceived, header.m_length - totalBytesReceived, 0);
        if (bytesReceived < 0)
        {
            std::cerr << "Error receiving JSON data. Error: " << strerror(errno) << std::endl;
            return false;
        }
        totalBytesReceived += bytesReceived;
    }
    std::cout << "Total bytes received=" << totalBytesReceived << std::endl;
    jsonDataBuf[header.m_length] = '\0';
    std::cout << "Received JSON data = " << jsonDataBuf << std::endl;
    jsonData = jsonDataBuf;

    delete[] jsonDataBuf;
    delete[] headerBuf;

    return true;
}
