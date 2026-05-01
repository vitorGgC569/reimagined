#include "dataloader.h"
#include <iostream>
#include <chrono>

DataLoader::DataLoader(const std::string& p, int bs, int dm, int mq)
    : path(p), batch_size(bs), d_model(dm), stop_flag(false), current_idx(0), total_samples(10000) {
    // In real app, open file 'path' and read header for total_samples
    worker = std::thread(&DataLoader::worker_loop, this);
}

DataLoader::~DataLoader() {
    stop_flag = true;
    not_full.notify_all();
    if (worker.joinable()) worker.join();
}

bool DataLoader::next(Tensor& batch) {
    std::unique_lock<std::mutex> lock(mutex);
    not_empty.wait(lock, [this]() { return !queue.empty() || stop_flag; });

    if (queue.empty() && stop_flag) return false;

    batch = std::move(queue.front());
    queue.pop();
    not_full.notify_one();
    return true;
}

#include <fstream>

void DataLoader::worker_loop() {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        // Fallback to random if file missing
        while (!stop_flag && current_idx < total_samples) {
             Tensor t = Tensor::random({batch_size, d_model}, Device::CPU);
             {
                std::unique_lock<std::mutex> lock(mutex);
                not_full.wait(lock, [this]() { return queue.size() < 5 || stop_flag; });
                if (stop_flag) break;
                queue.push(std::move(t));
                current_idx += batch_size;
             }
             not_empty.notify_one();
        }
        return;
    }

    // Real binary read loop
    while (!stop_flag && file.good()) {
        // Read batch_size * d_model floats
        // This assumes file is raw floats
        int num_floats = batch_size * d_model;

        // FIX: Zero-Copy read directly into Tensor memory
        Tensor t({batch_size, d_model}, Device::CPU);
        file.read(reinterpret_cast<char*>(t.data()), num_floats * sizeof(float));
        int read_count = file.gcount() / sizeof(float);

        if (read_count < num_floats) break; // EOF

        {
            std::unique_lock<std::mutex> lock(mutex);
            // FIX: Increase queue size to 16 to buffer IO latency
            not_full.wait(lock, [this]() { return queue.size() < 16 || stop_flag; });

            if (stop_flag) break;

            queue.push(std::move(t));
            current_idx += batch_size;
        }
        not_empty.notify_one();
    }
    stop_flag = true; // EOF
    not_empty.notify_all();
}
