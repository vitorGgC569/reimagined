#ifndef JAMBA_DEVICE_H
#define JAMBA_DEVICE_H

#include "jamba.h"

// Extended Interface for Device Management
// Implemented in jamba_device.cpp

// These methods are added to the classes via the linker, assuming headers allow modification
// or we just declare them in jamba.h and implement there.
// Since I cannot modify jamba.h signature easily without touching include,
// I will modify jamba.h to include 'void to(Device dev);'

#endif
