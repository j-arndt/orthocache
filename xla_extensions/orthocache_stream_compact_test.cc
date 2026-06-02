// Copyright 2026 OrthoCache Authors. All Rights Reserved.
//
// Unit tests for the OrthoCache Stream Compaction HLO Pass.

#include "orthocache_stream_compact.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = testing::opcode_matchers;

class OrthoCacheStreamCompactTest : public HloTestBase {};

// Constructs a minimal HLO module with an attention-like while loop:
//   init = (i32 loop_var, f32[M,D] accum, f32[N,D] kv_cache, pred[M] mask)
//   condition: loop_var < M
//   body: accum += dot(accum_slice, kv_slice); loop_var++
constexpr char kSimpleAttentionLoop[] = R"(
HloModule simple_attention

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

condition {
  state = (s32[], f32[1,128], f32[8192,128], pred[16]) parameter(0)
  loop_var = s32[] get-tuple-element(state), index=0
  limit = s32[] constant(16)
  ROOT cmp = pred[] compare(loop_var, limit), direction=LT
}

body {
  state = (s32[], f32[1,128], f32[8192,128], pred[16]) parameter(0)
  loop_var = s32[] get-tuple-element(state), index=0
  accum = f32[1,128] get-tuple-element(state), index=1
  kv = f32[8192,128] get-tuple-element(state), index=2
  mask = pred[16] get-tuple-element(state), index=3

  // Compute offset = loop_var * 512
  block_size = s32[] constant(512)
  offset = s32[] multiply(loop_var, block_size)
  zero = s32[] constant(0)

  // DynamicSlice: kv_block = kv[offset:offset+512, :]
  kv_block = f32[512,128] dynamic-slice(kv, offset, zero),
      dynamic_slice_sizes={512, 128}

  // Dot: partial = accum @ kv_block^T (simplified; real attention is Q·K^T)
  partial = f32[1,512] dot(accum, kv_block),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}

  // Reduce partial to update accumulator (simplified)
  zero_f32 = f32[] constant(0)
  reduced = f32[1] reduce(partial, zero_f32), dimensions={1}, to_apply=add_f32
  reshaped = f32[1,1] reshape(reduced)
  broadcast_reduced = f32[1,128] broadcast(reshaped), dimensions={0,1}
  new_accum = f32[1,128] add(accum, broadcast_reduced)

  // Increment loop var
  one = s32[] constant(1)
  new_loop_var = s32[] add(loop_var, one)

  ROOT new_state = (s32[], f32[1,128], f32[8192,128], pred[16])
      tuple(new_loop_var, new_accum, kv, mask)
}

ENTRY main {
  init_loop_var = s32[] constant(0)
  init_accum = f32[1,128] broadcast(f32[] constant(0)), dimensions={}
  kv_cache = f32[8192,128] parameter(0)
  block_mask = pred[16] parameter(1)

  init_state = (s32[], f32[1,128], f32[8192,128], pred[16])
      tuple(init_loop_var, init_accum, kv_cache, block_mask)

  ROOT result = (s32[], f32[1,128], f32[8192,128], pred[16])
      while(init_state), condition=condition, body=body
}
)";

TEST_F(OrthoCacheStreamCompactTest, MatchesAttentionLoop) {
  auto module = ParseAndReturnVerifiedModule(kSimpleAttentionLoop).value();

  OrthoCacheStreamCompact pass(/*block_size=*/512);
  auto result = pass.Run(module.get(), {});
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result.value());  // Module was changed
}

TEST_F(OrthoCacheStreamCompactTest, NoMatchOnNonAttentionLoop) {
  // A while-loop without any Dot → should not match
  constexpr char kNonAttentionLoop[] = R"(
  HloModule non_attention

  condition {
    state = (s32[], f32[10]) parameter(0)
    i = s32[] get-tuple-element(state), index=0
    limit = s32[] constant(10)
    ROOT cmp = pred[] compare(i, limit), direction=LT
  }

  body {
    state = (s32[], f32[10]) parameter(0)
    i = s32[] get-tuple-element(state), index=0
    arr = f32[10] get-tuple-element(state), index=1
    one = s32[] constant(1)
    new_i = s32[] add(i, one)
    ROOT new_state = (s32[], f32[10]) tuple(new_i, arr)
  }

  ENTRY main {
    init_i = s32[] constant(0)
    init_arr = f32[10] broadcast(f32[] constant(1)), dimensions={}
    init = (s32[], f32[10]) tuple(init_i, init_arr)
    ROOT result = (s32[], f32[10]) while(init), condition=condition, body=body
  }
  )";

  auto module = ParseAndReturnVerifiedModule(kNonAttentionLoop).value();
  OrthoCacheStreamCompact pass(/*block_size=*/512);
  auto result = pass.Run(module.get(), {});
  EXPECT_TRUE(result.ok());
  EXPECT_FALSE(result.value());  // Module was NOT changed
}

TEST_F(OrthoCacheStreamCompactTest, StreamCompactionProducesValidHLO) {
  auto module = ParseAndReturnVerifiedModule(kSimpleAttentionLoop).value();

  OrthoCacheStreamCompact pass(/*block_size=*/512);
  auto result = pass.Run(module.get(), {});
  ASSERT_TRUE(result.ok());

  // After the pass, the module should still verify
  auto status = module->Verify();
  EXPECT_TRUE(status.ok()) << status;
}

}  // namespace
}  // namespace xla
