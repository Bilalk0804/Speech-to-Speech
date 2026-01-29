#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <cstddef>
#include <cstring>

template<typename T>
class RingBuffer {
public:
    RingBuffer(size_t capacity) : capacity(capacity), write_pos(0), read_pos(0) {
        buffer = new T[capacity];
    }
    
    ~RingBuffer() {
        delete[] buffer;
    }
    
    size_t write(const T* src, size_t count);
    size_t read(T* dst, size_t count);
    size_t available() const;
    size_t space() const;
    
private:
    T* buffer;
    size_t capacity;
    size_t write_pos;
    size_t read_pos;
};

#endif // RING_BUFFER_H
