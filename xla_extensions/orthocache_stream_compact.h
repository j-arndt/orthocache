#ifndef XLA_SERVICE_TPU_ORTHOCACHE_STREAM_COMPACT_H_
#define XLA_SERVICE_TPU_ORTHOCACHE_STREAM_COMPACT_H_

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// HLO Pass that rewrites attention while-loops into dynamically bounded loops
// using a CustomCall prefix-sum for stream compaction.
//
// This breaks MXU predication on TPU by dynamically shortening the trip count
// of the attention loop based on the number of active (non-evicted) blocks.
class OrthoCacheStreamCompact : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "orthocache-stream-compact";
  }

  // Run the pass on the given HLO module.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Helper to identify if a while-loop represents the dense attention block loop.
  bool IsAttentionLoop(HloInstruction* while_inst);

  // Helper to rewrite the while-loop.
  absl::StatusOr<bool> RewriteAttentionLoop(HloInstruction* while_inst);
};

}  // namespace xla

#endif  // XLA_SERVICE_TPU_ORTHOCACHE_STREAM_COMPACT_H_
