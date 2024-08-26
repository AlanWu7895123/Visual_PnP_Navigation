#include "agv_system.h"

int main()
{
    int mode;
    cout << "choose mode, 0 is test, 1 is normal\n";
    cin >> mode;

    AGVSystem agvSystem(mode);
    agvSystem.setTarget(1, 1);
    agvSystem.run();
    return 0;
}