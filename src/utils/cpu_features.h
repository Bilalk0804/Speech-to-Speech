#ifndef CPU_FEATURES_H
#define CPU_FEATURES_H

class CPUFeatures {
public:
    static bool hasNEON();
    static bool hasSME2();
    static bool hasAVX2();
    static bool hasSVE();
    
    static int getCoreCount();
    static int getMaxFrequency();
    
private:
    static bool neon_detected;
    static bool sme2_detected;
};

#endif // CPU_FEATURES_H
