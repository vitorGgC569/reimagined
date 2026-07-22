#include <iostream>

// Compatibility tombstone for the pre-SDK demonstration server.  The old
// implementation bound INADDR_ANY without authentication, parsed requests from
// a fixed 1 KiB buffer and returned fabricated inference output.  Keeping that
// code in-tree made accidental manual builds unsafe even though CMake no longer
// referenced it.  Production serving lives in api_server.cpp and
// http_api_server.cpp.
int main() {
    std::cerr << "This legacy server has been retired. Build and run "
                 "the 'nsos_api_server' target instead.\n";
    return 2;
}
