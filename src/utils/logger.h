#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>

enum LogLevel {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARNING = 2,
    LOG_ERROR = 3
};

class Logger {
public:
    static void log(LogLevel level, const char* message);
    static void setLogLevel(LogLevel level);
    
private:
    static LogLevel current_level;
};

#endif // LOGGER_H
