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
  {
    // batch_size=2500 reaches the mock fallback's 10,000-sample bound in four
    // batches.  The fifth next() must observe completion instead of waiting
    // forever.
    DataLoader loader(missing_path.string(), 2500, 1, 1);
    nsos::Tensor batch;
    int batches = 0;
    while (loader.next(batch)) {
      ++batches;
    }
    assert(batches == 4);
  }

  bool invalid_rejected = false;
  try {
    DataLoader invalid(missing_path.string(), 0, 1, 1);
    (void)invalid;
  } catch (const std::invalid_argument&) {
    invalid_rejected = true;
  }
  assert(invalid_rejected);

  std::cout << "DataLoader termination and bounds test passed!\n";
  return 0;
}
