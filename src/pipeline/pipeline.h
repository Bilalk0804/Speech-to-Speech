#ifndef PIPELINE_H
#define PIPELINE_H

class Pipeline {
public:
    Pipeline();
    ~Pipeline();
    
    bool initialize();
    void process();
    void shutdown();
    
private:
    bool initialized;
};

#endif // PIPELINE_H
