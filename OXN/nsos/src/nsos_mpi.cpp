#include "../include/nsos_mpi.h"
#include <iostream>

namespace nsos {

void MpiManager::init(int* argc, char*** argv) {
    std::cout << "[NSOS] MPI Disabled: Running Single Node Mode" << std::endl;
}

void MpiManager::finalize() {}

int MpiManager::rank() {
    return 0;
}

int MpiManager::size() {
    return 1;
}

void MpiManager::barrier() {}

} // namespace nsos
