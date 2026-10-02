#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <array>

namespace nsos::gpu {

enum class WorkspaceSlot : unsigned {
  BitnetActivations, BitnetScales, BitnetLut, MoePrepared, MoeScales, MoeHidden,
  MoeContributions, GroupedPrepared, GroupedScales, AttentionPartials, Count
};
enum class StorageType : unsigned { Int8, Int16, Float32, UInt32, Float16, Bytes };
enum class DispatchPath : unsigned {
  BitnetGemv, BitnetGemm, SparseMoe, GroupedProjection, TiledAttention,
  CompactAttention, MambaEpilogue, GraphReplay, GroupedMoeTraining, GroupedMoeWmmaGemm,
  GroupedMoeGradientCommit, KanRecompute, KanWmmaGemm, Mamba3SisoForward, Mamba3SisoBackward,
  Mamba3PreprocessForward, Mamba3PreprocessBackward, Mamba3ProjectionWmma, Count
};
void record_dispatch(DispatchPath path) noexcept;
std::array<std::uint64_t, static_cast<unsigned>(DispatchPath::Count)> dispatch_counters() noexcept;

// A serialized execution lane, not a session or a weight owner. All scratch is
// private to this context and remains alive until queued work completes.
class ExecutionContext {
 public:
  ExecutionContext();
  ~ExecutionContext();
  ExecutionContext(const ExecutionContext&) = delete;
  ExecutionContext& operator=(const ExecutionContext&) = delete;
  void* reserve(WorkspaceSlot slot, StorageType type, std::size_t elements);
  std::size_t reserved_bytes() const noexcept;
  // During graph capture, growing a workspace would invalidate captured
  // pointers. Warm every shape first, then freeze; failure is explicit.
  void freeze(bool enabled) noexcept;

  // Serializes classic BLAS host calls with the lane. The handle is created
  // lazily, bound once to this lane's device/stream, and never migrated.
  // Opaque vendor handle keeps the public/CPU layout independent of SDK types.
  class ClassicBlasLease {
   public:
    explicit ClassicBlasLease(ExecutionContext& context);
    ~ClassicBlasLease();
    ClassicBlasLease(const ClassicBlasLease&) = delete;
    ClassicBlasLease& operator=(const ClassicBlasLease&) = delete;
    void* handle() const noexcept { return handle_; }
   private:
    ExecutionContext* context_ = nullptr;
    void* handle_ = nullptr;
  };

  class Scope {
   public:
    explicit Scope(ExecutionContext& context, bool enabled = true);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
   private:
    struct State;
    std::unique_ptr<State> state_;
  };
 private:
  explicit ExecutionContext(bool default_lane);
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class Scope;
  friend ExecutionContext& current_execution_context();
};

// Standalone operators use a thread-local lane; model calls bind their own.
ExecutionContext& current_execution_context();

}  // namespace nsos::gpu
