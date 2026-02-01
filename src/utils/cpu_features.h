// simple example (Linux) using getauxval for HWCAP
#include <sys/auxv.h>
#include <stdint.h>
#include <stdbool.h>

static inline bool cpu_has_neon() {
    #ifdef HWCAP_NEON
    unsigned long hwcap = getauxval(AT_HWCAP);
    return (hwcap & HWCAP_NEON) != 0;
    #else
    return false;
    #endif
}
