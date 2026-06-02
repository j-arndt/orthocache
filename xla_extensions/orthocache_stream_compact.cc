#include "orthocache_stream_compact.h"

#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/pattern_matcher.h"

namespace xla {

namespace m = match;

bool OrthoCacheStreamCompact::IsAttentionLoop(HloInstruction* while_inst) {
  // A heuristic to match the Pallas sparse attention block loop.
  // In a real implementation, we would match on specific CustomCalls or
  // dot operations inside the while body computation.
  auto* body = while_inst->while_body();
  bool has_dot = false;
  for (auto* inst : body->instructions()) {
    if (inst->opcode() == HloOpcode::kDot) {
      has_dot = true;
      break;
    }
  }
  return has_dot;
}

absl::StatusOr<bool> OrthoCacheStreamCompact::RewriteAttentionLoop(HloInstruction* while_inst) {
  // 1. We intercept the block_mask from the input of the while loop or the surrounding graph.
  // 2. We inject a CustomCall to "orthocache_stream_compact" that takes block_mask
  //    and returns (active_indices, num_active).
  // 3. We rewrite the loop condition to terminate when loop_var >= num_active.
  // 4. We rewrite the loop body to load blocks using dynamic-slice with index active_indices[loop_var].
  
  // This is a stub implementation representing the transformation structure.
  return true;
}

absl::StatusOr<bool> OrthoCacheStreamCompact::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    std::vector<HloInstruction*> while_loops;
    for (HloInstruction* inst : computation->instructions()) {
      if (inst->opcode() == HloOpcode::kWhile && IsAttentionLoop(inst)) {
        while_loops.push_back(inst);
      }
    }

    for (HloInstruction* while_inst : while_loops) {
      absl::StatusOr<bool> result = RewriteAttentionLoop(while_inst);
      if (result.ok() && result.value()) {
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xla
