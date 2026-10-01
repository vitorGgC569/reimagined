#ifndef LEAN_INTEGRATION_H
#define LEAN_INTEGRATION_H

#include <stdexcept>
#include <string>

namespace nsos {

/**
 * Fail-closed boundary for an optional Lean 4 verification provider.
 *
 * No solver is linked in this build. In particular, this class must never
 * infer validity from syntax or return an optimistic result: callers can test
 * availability and must treat an unavailable provider as unverified.
 */
class LeanVerifier {
public:
  LeanVerifier() = default;

  [[nodiscard]] constexpr bool available() const noexcept { return false; }

  /**
   * verify: Deterministic verification of logical/mathematical statements.
   */
  [[nodiscard]] bool verify(const std::string &statement) const {
    if (statement.empty()) {
      throw std::invalid_argument(
          "Lean verification requires a non-empty statement");
    }
    throw std::runtime_error(
        "Lean verification is unavailable: this NSOS build has no "
        "authenticated Lean 4 provider");
  }
};

} // namespace nsos

#endif // LEAN_INTEGRATION_H
