#include "../include/dataloader_v2.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  const auto real_path =
      std::filesystem::temp_directory_path() / "nsos_dataloader_test.bin";
  {
    const std::vector<float> values(12, 1.25f);
    std::ofstream output(real_path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
  }

  {
    DataLoader loader(real_path.string(), 2, 3, 1);
    nsos::Tensor batch;
    int batches = 0;
    while (loader.next(batch)) {
      assert(batch.shape.dims == std::vector<int>({2, 3}));
      assert(batch.data()[0] == 1.25f);
      ++batches;
    }
    assert(batches == 2);
  }
  std::error_code error;
  std::filesystem::remove(real_path, error);

  const auto missing_path =
      std::filesystem::temp_directory_path() / "nsos_missing_dataloader.bin";
  std::filesystem::remove(missing_path, error);
  bool missing_rejected = false;
  try {
    DataLoader missing(missing_path.string(), 2500, 1, 1);
    (void)missing;
  } catch (const std::runtime_error&) {
    missing_rejected = true;
  }
  assert(missing_rejected);

  const auto truncated_path =
      std::filesystem::temp_directory_path() / "nsos_truncated_dataloader.bin";
  {
    const std::vector<float> values(5, 2.0f);
    std::ofstream output(truncated_path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
  }
  bool truncated_rejected = false;
  try {
    DataLoader truncated(truncated_path.string(), 2, 3, 1);
    (void)truncated;
  } catch (const std::runtime_error&) {
    truncated_rejected = true;
  }
  assert(truncated_rejected);
  std::filesystem::remove(truncated_path, error);

  const auto empty_path =
      std::filesystem::temp_directory_path() / "nsos_empty_dataloader.bin";
  {
    std::ofstream output(empty_path, std::ios::binary | std::ios::trunc);
  }
  bool empty_rejected = false;
  try {
    DataLoader empty(empty_path.string(), 1, 1, 1);
    (void)empty;
  } catch (const std::runtime_error&) {
    empty_rejected = true;
  }
  assert(empty_rejected);
  std::filesystem::remove(empty_path, error);

  bool invalid_rejected = false;
  try {
    DataLoader invalid(missing_path.string(), 0, 1, 1);
    (void)invalid;
  } catch (const std::invalid_argument&) {
    invalid_rejected = true;
  }
  assert(invalid_rejected);

  std::cout << "DataLoader integrity, termination and bounds test passed!\n";
  return 0;
}
