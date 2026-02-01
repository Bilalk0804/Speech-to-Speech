#include <iostream>
#include "realtime_manager.h"

int main(int argc, char* argv[]) {
    std::cout << "S2S On-Device Application" << std::endl;
    
    RealtimeManager manager;
    manager.initialize();
    
    return 0;
}
