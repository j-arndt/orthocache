// Copyright 2026 OrthoCache Authors. All Rights Reserved.
//
// XLA HLO Loop-Reindexing Pass for OrthoCache.
//
// This pass transforms predicated attention loops into compacted loops
// that iterate only over non-evicted KV blocks. It injects a stream
// compaction primitive (parallel prefix sum) before the attention loop,
// then rewrites the loop's trip count and body to use indirect indexing.
//
// Pass location in the XLA pipeline:
//   After algebraic simplification, before buffer assignment.
//
// Fallback: If pattern matching fails, the module is returned unchanged
//   (dense execution, no regression).

#ifndef XLA_SERVICE_ORTHOCACHE_STREAM_COMPACT_H_
#define XLA_SERVICE_ORTHOCACHE_STREAM_COMPACT_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Describes a matched attention loop and its relevant operands.
struct AttentionLoopMatch {
  HloInstruction* while_inst;      // The kWhile instruction
  HloInstruction* block_mask;      // The boolean mask operand
  int mask_tuple_index;            // Index of mask in the while init tuple
  int kv_tuple_index_k;            // Index of K cache in the while init tuple
  int kv_tuple_index_v;            // Index of V cache in the while init tuple
  int loop_var_index;              // Index of the induction variable in tuple
  int64_t num_blocks;              // Static trip count (M)
  int64_t block_size;              // Block size in tokens
};

class OrthoCacheStreamCompact : public HloModulePass {
 public:
  explicit OrthoCacheStreamCompact(
      int64_t block_size = 512,
      float max_eviction_rate = 0.70f)
      : block_size_(block_size),
        max_eviction_rate_(max_eviction_rate) {}

  absl::string_view name() const override {
    return "orthocache-stream-compact";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Stage 0: Pattern-match attention while-loops in the HLO graph.
  // Returns true and populates `match` if a valid attention loop is found.
  bool MatchAttentionLoop(HloInstruction* while_inst,
                          AttentionLoopMatch* match);

  // Stage 1: Inject stream compaction (prefix sum + popcount) before the loop.
  // Returns the (active_indices, num_active) instructions.
  absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>>
  InjectStreamCompaction(HloComputation* computation,
                         HloInstruction* block_mask,
                         int64_t num_blocks);

  // Stage 2: Rewrite the while-loop condition to use dynamic trip count.
  absl::Status RewriteLoopCondition(HloInstruction* while_inst,
                                    HloInstruction* num_active,
                                    const AttentionLoopMatch& match);

  // Stage 3: Rewrite the while-loop body to use indirect DMA via
  // the active_indices indirection table.
  absl::Status RewriteLoopBody(HloInstruction* while_inst,
                               HloInstruction* active_indices,
                               const AttentionLoopMatch& match);

  int64_t block_size_;
  float max_eviction_rate_;
};

}  // namespace xla

#endif  // XLA_SERVICE_ORTHOCACHE_STREAM_COMPACT_H_
