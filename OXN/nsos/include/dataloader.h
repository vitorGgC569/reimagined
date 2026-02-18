#ifndef DATALOADER_H
#define DATALOADER_H

#include "tensor.h"
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>

class DataLoader {
public:
    DataLoader(const std::string& bin_path, int batch_size, int d_model, int max_queue=5);
    ~DataLoader();

    // Returns true if a batch was loaded, false if EOF
    bool next(Tensor& batch);

private:
    void worker_loop();

    std::string path;
    int batch_size;
    int d_model;

    std::thread worker;
    std::atomic<bool> stop_flag;

    std::queue<Tensor> queue;
    std::mutex mutex;
    std::condition_variable not_empty;
    std::condition_variable not_full;

    // Mock file pointer
    size_t current_idx;
    size_t total_samples;
};

#endif
