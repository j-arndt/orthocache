// Copyright 2026 OrthoCache Authors. All Rights Reserved.
//
// Implementation of the OrthoCache Stream Compaction HLO Pass.
//
// Three-stage transformation:
//   1. Inject prefix_sum(block_mask) → active_indices[K], num_active
//   2. Rewrite while-loop condition: i < M  →  i < num_active
//   3. Rewrite while-loop body: DMA[i] → DMA[active_indices[i]]
//
// The net effect: the MXU only fires for retained blocks.
// Wall-clock savings ≈ S × T_attention, where S is eviction fraction.

#include "orthocache_stream_compact.h"

#include <algorithm>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"

namespace xla {

namespace m = match;

// ============================================================================
// Stage 0: Pattern Matching
// ============================================================================

bool OrthoCacheStreamCompact::MatchAttentionLoop(
    HloInstruction* while_inst,
    AttentionLoopMatch* match) {
  if (while_inst->opcode() != HloOpcode::kWhile) return false;

  HloComputation* body = while_inst->while_body();
  HloComputation* cond = while_inst->while_condition();

  // The body must contain at least one kDot (the Q·K^T matmul).
  bool has_dot = false;
  bool has_dynamic_slice = false;
  for (auto* inst : body->instructions()) {
    if (inst->opcode() == HloOpcode::kDot) {
      has_dot = true;
    }
    if (inst->opcode() == HloOpcode::kDynamicSlice) {
      has_dynamic_slice = true;
    }
  }

  if (!has_dot) {
    VLOG(2) << "OrthoCacheStreamCompact: Skipping while-loop at "
            << while_inst->name() << " — no Dot in body.";
    return false;
  }

  // The init must be a tuple (induction_var, accumulators..., kv_cache..., mask).
  HloInstruction* init = while_inst->mutable_operand(0);
  if (init->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "OrthoCacheStreamCompact: Skipping — init is not a tuple.";
    return false;
  }

  // Search the tuple elements for:
  //  - A scalar S32 (the loop induction variable)
  //  - A 1-D bool array (the block mask)
  //  - 3-D F32/BF16 arrays (K and V caches)
  int loop_var_idx = -1;
  int mask_idx = -1;
  int k_idx = -1;
  int v_idx = -1;

  for (int i = 0; i < init->operand_count(); ++i) {
    const Shape& shape = init->operand(i)->shape();

    // Scalar integer → induction variable candidate
    if (shape.rank() == 0 && shape.element_type() == S32) {
      if (loop_var_idx == -1) loop_var_idx = i;
    }

    // 1-D bool → block mask candidate
    if (shape.rank() == 1 && shape.element_type() == PRED) {
      mask_idx = i;
    }

    // 2-D bool → block mask with head dim (num_blocks, num_heads)
    if (shape.rank() == 2 && shape.element_type() == PRED) {
      mask_idx = i;
    }

    // 3-D float → KV cache candidate (seq_len, num_heads, head_dim)
    if (shape.rank() == 3 &&
        (shape.element_type() == F32 || shape.element_type() == BF16)) {
      if (k_idx == -1) {
        k_idx = i;
      } else if (v_idx == -1) {
        v_idx = i;
      }
    }
  }

  if (loop_var_idx == -1 || mask_idx == -1 || k_idx == -1) {
    VLOG(2) << "OrthoCacheStreamCompact: Could not identify loop structure "
            << "in while-loop " << while_inst->name();
    return false;
  }

  // Extract the static trip count from the condition computation.
  // The condition should be: get-tuple-element(param, loop_var_idx) < const
  int64_t trip_count = -1;
  HloInstruction* cond_root = cond->root_instruction();
  if (cond_root->opcode() == HloOpcode::kCompare) {
    HloInstruction* rhs = cond_root->mutable_operand(1);
    if (rhs->opcode() == HloOpcode::kConstant) {
      auto maybe_val = rhs->literal().GetFirstInteger();
      if (maybe_val.has_value()) {
        trip_count = *maybe_val;
      }
    }
  }

  if (trip_count <= 0) {
    VLOG(2) << "OrthoCacheStreamCompact: Could not extract trip count from "
            << "condition of " << while_inst->name();
    return false;
  }

  // Populate the match struct.
  match->while_inst = while_inst;
  match->block_mask = init->mutable_operand(mask_idx);
  match->mask_tuple_index = mask_idx;
  match->kv_tuple_index_k = k_idx;
  match->kv_tuple_index_v = v_idx;
  match->loop_var_index = loop_var_idx;
  match->num_blocks = trip_count;
  match->block_size = block_size_;

  VLOG(1) << "OrthoCacheStreamCompact: Matched attention loop "
          << while_inst->name() << " with " << trip_count << " blocks, "
          << "mask at tuple index " << mask_idx;

  return true;
}

// ============================================================================
// Stage 1: Stream Compaction (Prefix Sum + Popcount)
// ============================================================================

absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>>
OrthoCacheStreamCompact::InjectStreamCompaction(
    HloComputation* computation,
    HloInstruction* block_mask,
    int64_t num_blocks) {

  // If block_mask is 2-D (num_blocks, num_heads), reduce to 1-D by taking
  // logical-OR across heads. Any block retained by ANY head stays active.
  HloInstruction* mask_1d = block_mask;
  if (block_mask->shape().rank() == 2) {
    // Reduce-or across the head dimension (axis 1)
    auto reduce_shape = ShapeUtil::MakeShape(PRED, {num_blocks});
    HloComputation::Builder or_builder("or_reduce");
    auto or_lhs = or_builder.AddInstruction(
        HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(PRED, {}), "a"));
    auto or_rhs = or_builder.AddInstruction(
        HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(PRED, {}), "b"));
    or_builder.AddInstruction(
        HloInstruction::CreateBinary(ShapeUtil::MakeShape(PRED, {}),
                                     HloOpcode::kOr, or_lhs, or_rhs));
    HloComputation* or_comp =
        computation->parent()->AddEmbeddedComputation(or_builder.Build());

    auto init_val = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));

    mask_1d = computation->AddInstruction(
        HloInstruction::CreateReduce(reduce_shape, block_mask, init_val,
                                     {1}, or_comp));
  }

  // Convert bool mask to int32 for arithmetic: mask_i32[i] = mask[i] ? 1 : 0
  auto i32_shape = ShapeUtil::MakeShape(S32, {num_blocks});
  auto ones = computation->AddInstruction(
      HloInstruction::CreateBroadcast(
          i32_shape,
          computation->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1))),
          {}));
  auto zeros = computation->AddInstruction(
      HloInstruction::CreateBroadcast(
          i32_shape,
          computation->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0))),
          {}));

  // Broadcast mask_1d to i32_shape for the select
  auto mask_broadcast = mask_1d;
  if (mask_1d->shape().element_type() != PRED) {
    // Already PRED, no conversion needed
  }

  auto mask_i32 = computation->AddInstruction(
      HloInstruction::CreateTernary(i32_shape, HloOpcode::kSelect,
                                    mask_broadcast, ones, zeros));

  // --- Popcount: num_active = sum(mask_i32) ---
  HloComputation::Builder add_builder("add_reduce");
  auto add_lhs = add_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "a"));
  auto add_rhs = add_builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(S32, {}), "b"));
  add_builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(S32, {}),
                                   HloOpcode::kAdd, add_lhs, add_rhs));
  HloComputation* add_comp =
      computation->parent()->AddEmbeddedComputation(add_builder.Build());

  auto zero_scalar = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));

  auto num_active = computation->AddInstruction(
      HloInstruction::CreateReduce(ShapeUtil::MakeShape(S32, {}),
                                   mask_i32, zero_scalar, {0}, add_comp));

  // --- Prefix Sum: cumulative_sum(mask_i32) ---
  // XLA doesn't have a native cumsum. We implement it as a ReduceWindow
  // with a causal window [1, 1, ..., 1] of width num_blocks and stride 1.
  //
  // Alternatively, for small num_blocks (≤256), we use a custom-call to
  // the TPU VPU prefix sum. For portability, we use ReduceWindow here.
  Window window;
  auto* dim = window.add_dimensions();
  dim->set_size(num_blocks);
  dim->set_stride(1);
  dim->set_padding_low(num_blocks - 1);
  dim->set_padding_high(0);
  dim->set_window_dilation(1);
  dim->set_base_dilation(1);

  auto prefix_sum = computation->AddInstruction(
      HloInstruction::CreateReduceWindow(
          i32_shape, mask_i32, zero_scalar, window, add_comp));

  // --- Build active_indices from prefix_sum ---
  // active_indices[prefix_sum[i] - 1] = i, for all i where mask[i] == 1
  //
  // This is a scatter operation. But for the HLO pass, we can use a simpler
  // approach: sort the indices by (1-mask, original_index) so active blocks
  // come first in their original order.
  //
  // iota = [0, 1, 2, ..., M-1]
  auto iota = computation->AddInstruction(
      HloInstruction::CreateIota(i32_shape, 0));

  // Sort key: inactive blocks get key = M + original_index (pushed to end)
  //           active blocks get key = original_index (stay in order)
  // sort_key[i] = mask[i] ? i : (M + i)
  auto m_const = computation->AddInstruction(
      HloInstruction::CreateBroadcast(
          i32_shape,
          computation->AddInstruction(
              HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(num_blocks))),
          {}));
  auto inactive_keys = computation->AddInstruction(
      HloInstruction::CreateBinary(i32_shape, HloOpcode::kAdd, iota, m_const));
  auto sort_keys = computation->AddInstruction(
      HloInstruction::CreateTernary(i32_shape, HloOpcode::kSelect,
                                    mask_broadcast, iota, inactive_keys));

  // Sort (sort_keys, iota) by sort_keys ascending.
  // The first K entries of the sorted iota are the active indices.
  HloComputation::Builder cmp_builder("sort_comparator");
  auto cmp_lhs_key = cmp_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "lk"));
  auto cmp_rhs_key = cmp_builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(S32, {}), "rk"));
  // Parameters 2,3 are for the payload (iota values)
  cmp_builder.AddInstruction(
      HloInstruction::CreateParameter(2, ShapeUtil::MakeShape(S32, {}), "lv"));
  cmp_builder.AddInstruction(
      HloInstruction::CreateParameter(3, ShapeUtil::MakeShape(S32, {}), "rv"));
  cmp_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}),
                                    cmp_lhs_key, cmp_rhs_key,
                                    ComparisonDirection::kLt));
  HloComputation* cmp_comp =
      computation->parent()->AddEmbeddedComputation(cmp_builder.Build());

  auto sort = computation->AddInstruction(
      HloInstruction::CreateSort(
          ShapeUtil::MakeTupleShape({i32_shape, i32_shape}),
          /*dimension=*/0,
          {sort_keys, iota},
          cmp_comp,
          /*is_stable=*/true));

  // Extract the sorted indices (second element of the sort output tuple)
  auto active_indices = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(i32_shape, sort, 1));

  VLOG(1) << "OrthoCacheStreamCompact: Injected stream compaction for "
          << num_blocks << " blocks";

  return std::make_pair(active_indices, num_active);
}

// ============================================================================
// Stage 2: Rewrite Loop Condition
// ============================================================================

absl::Status OrthoCacheStreamCompact::RewriteLoopCondition(
    HloInstruction* while_inst,
    HloInstruction* num_active,
    const AttentionLoopMatch& match) {

  HloComputation* old_cond = while_inst->while_condition();
  HloInstruction* cond_root = old_cond->root_instruction();

  // The condition should be: GTE(param, loop_var_idx) < constant(M)
  // We replace constant(M) with num_active.
  //
  // Since the condition is an embedded computation, we need to:
  // 1. Add num_active as a new element in the while-loop's init tuple
  // 2. Extract it inside the condition computation
  // 3. Replace the static constant comparison

  // For now, we take a simpler approach: we replace the constant in the
  // condition with a dynamic value passed through the tuple.

  // Find the constant operand in the comparison
  if (cond_root->opcode() != HloOpcode::kCompare) {
    return absl::InternalError(
        "Expected kCompare as condition root, got " +
        HloOpcodeString(cond_root->opcode()));
  }

  // The RHS of the comparison is the trip count constant.
  // We need to thread num_active through the while-loop tuple.
  // This requires modifying the tuple shape, init, condition, and body.

  // Step 1: Extend the init tuple to include num_active
  HloInstruction* old_init = while_inst->mutable_operand(0);
  std::vector<HloInstruction*> new_init_elements;
  for (int i = 0; i < old_init->operand_count(); ++i) {
    new_init_elements.push_back(old_init->mutable_operand(i));
  }
  int num_active_idx = new_init_elements.size();
  new_init_elements.push_back(num_active);

  // Build new tuple shape
  std::vector<Shape> new_shapes;
  for (auto* elem : new_init_elements) {
    new_shapes.push_back(elem->shape());
  }
  Shape new_tuple_shape = ShapeUtil::MakeTupleShape(new_shapes);

  auto new_init = old_init->parent()->AddInstruction(
      HloInstruction::CreateTuple(new_init_elements));

  // Step 2: Clone the condition computation with the new tuple parameter
  // and replace the constant with GTE(param, num_active_idx)
  HloComputation::Builder new_cond_builder("reindexed_condition");
  auto new_cond_param = new_cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, new_tuple_shape, "loop_state"));

  auto loop_var_gte = new_cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(
          ShapeUtil::MakeShape(S32, {}), new_cond_param,
          match.loop_var_index));

  auto dynamic_bound = new_cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(
          ShapeUtil::MakeShape(S32, {}), new_cond_param,
          num_active_idx));

  new_cond_builder.AddInstruction(
      HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}),
          loop_var_gte, dynamic_bound,
          ComparisonDirection::kLt));

  HloComputation* new_cond =
      while_inst->GetModule()->AddEmbeddedComputation(
          new_cond_builder.Build());

  // Step 3: Clone the body computation with the new tuple parameter
  // The body must also output the extended tuple (pass num_active through)
  HloComputation* old_body = while_inst->while_body();
  HloComputation* new_body = old_body;
  // TODO: Extend body to handle the new tuple element.
  // For now, we handle this in RewriteLoopBody.

  // Step 4: Update the while instruction
  while_inst->set_while_condition(new_cond);
  TF_RETURN_IF_ERROR(
      while_inst->ReplaceOperandWith(0, new_init));

  VLOG(1) << "OrthoCacheStreamCompact: Rewrote loop condition, "
          << "num_active at tuple index " << num_active_idx;

  return absl::OkStatus();
}

// ============================================================================
// Stage 3: Rewrite Loop Body (Indirect Indexing)
// ============================================================================

absl::Status OrthoCacheStreamCompact::RewriteLoopBody(
    HloInstruction* while_inst,
    HloInstruction* active_indices,
    const AttentionLoopMatch& match) {

  HloComputation* body = while_inst->while_body();

  // Find all DynamicSlice instructions in the body that use the loop
  // induction variable to index into the KV cache.
  //
  // Pattern: dynamic-slice(kv_cache, loop_var * block_size, ...)
  //
  // We replace the direct index with: active_indices[loop_var] * block_size

  // First, add active_indices to the while-loop tuple (similar to num_active).
  // The active_indices tensor is passed through the loop unchanged.

  // Find the parameter instruction in the body
  HloInstruction* body_param = body->parameter_instruction(0);

  // Find GTE(param, loop_var_index) — this is the current loop variable
  HloInstruction* loop_var_gte = nullptr;
  for (auto* user : body_param->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == match.loop_var_index) {
      loop_var_gte = user;
      break;
    }
  }

  if (loop_var_gte == nullptr) {
    return absl::InternalError(
        "Could not find loop variable GTE in body of " +
        while_inst->name());
  }

  // For each DynamicSlice that uses the loop variable (directly or through
  // multiplication), replace the index computation with indirect lookup.
  //
  // Before: offset = loop_var * block_size
  // After:  real_block = active_indices[loop_var]
  //         offset = real_block * block_size

  // We need active_indices accessible inside the body.
  // It's at the last index of the extended tuple (added alongside num_active).
  // TODO: Coordinate with RewriteLoopCondition to know the exact index.

  // For each user of loop_var_gte that is a multiply (offset computation):
  for (auto* user : loop_var_gte->users()) {
    if (user->opcode() == HloOpcode::kMultiply) {
      // This is likely: loop_var * block_size
      // Replace loop_var with active_indices[loop_var]

      // Create: dynamic-slice(active_indices, {loop_var}, {1})
      auto index_lookup = body->AddInstruction(
          HloInstruction::CreateDynamicSlice(
              ShapeUtil::MakeShape(S32, {1}),
              body_param,  // TODO: this should be GTE for active_indices
              {loop_var_gte},
              {1}));

      auto index_scalar = body->AddInstruction(
          HloInstruction::CreateReshape(
              ShapeUtil::MakeShape(S32, {}), index_lookup));

      // Replace the loop_var operand in the multiply with the looked-up index
      TF_RETURN_IF_ERROR(user->ReplaceOperandWith(
          user->operand_index(loop_var_gte), index_scalar));

      VLOG(2) << "OrthoCacheStreamCompact: Replaced direct index with "
              << "indirect lookup in " << user->name();
    }
  }

  VLOG(1) << "OrthoCacheStreamCompact: Rewrote loop body for indirect "
          << "indexing in " << while_inst->name();

  return absl::OkStatus();
}

// ============================================================================
// Main Pass Entry Point
// ============================================================================

absl::StatusOr<bool> OrthoCacheStreamCompact::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {

  VLOG(1) << "OrthoCacheStreamCompact: Running on module " << module->name();

  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    if (computation->IsFusionComputation()) continue;

    // Collect while-loops first (avoid mutating while iterating)
    std::vector<HloInstruction*> while_loops;
    for (HloInstruction* inst : computation->instructions()) {
      if (inst->opcode() == HloOpcode::kWhile) {
        while_loops.push_back(inst);
      }
    }

    for (HloInstruction* while_inst : while_loops) {
      AttentionLoopMatch match;
      if (!MatchAttentionLoop(while_inst, &match)) continue;

      VLOG(1) << "OrthoCacheStreamCompact: Processing " << while_inst->name()
              << " (" << match.num_blocks << " blocks)";

      // Stage 1: Inject stream compaction
      auto compaction_result = InjectStreamCompaction(
          computation, match.block_mask, match.num_blocks);
      if (!compaction_result.ok()) {
        LOG(WARNING) << "OrthoCacheStreamCompact: Failed to inject compaction "
                     << "for " << while_inst->name() << ": "
                     << compaction_result.status();
        continue;
      }

      auto [active_indices, num_active] = *compaction_result;

      // Stage 2: Rewrite loop condition
      auto cond_status = RewriteLoopCondition(while_inst, num_active, match);
      if (!cond_status.ok()) {
        LOG(WARNING) << "OrthoCacheStreamCompact: Failed to rewrite condition "
                     << "for " << while_inst->name() << ": " << cond_status;
        continue;
      }

      // Stage 3: Rewrite loop body
      auto body_status = RewriteLoopBody(while_inst, active_indices, match);
      if (!body_status.ok()) {
        LOG(WARNING) << "OrthoCacheStreamCompact: Failed to rewrite body "
                     << "for " << while_inst->name() << ": " << body_status;
        continue;
      }

      changed = true;
    }
  }

  if (changed) {
    VLOG(1) << "OrthoCacheStreamCompact: Module modified.";
  } else {
    VLOG(1) << "OrthoCacheStreamCompact: No attention loops matched.";
  }

  return changed;
}

}  // namespace xla
