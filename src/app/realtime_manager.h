#ifndef REALTIME_MANAGER_H
#define REALTIME_MANAGER_H

class RealtimeManager {
public:
    RealtimeManager();
    ~RealtimeManager();
    
    void initialize();
    void run();
    void shutdown();
    
private:
    bool initialized;
};

#endif // REALTIME_MANAGER_H
