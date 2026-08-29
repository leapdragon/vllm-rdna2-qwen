#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

// skinny_gemms.cu (LLMM1/wvSplitK/wvSplitKrc/wvSplitKQ) is excluded on gfx1250
// (gfx9/gfx11 ISA, unsupported there); skip these registrations to avoid
// undefined symbols. vLLM uses default/Triton GEMM for these ops on gfx1250.
#ifndef VLLM_SKIP_SKINNY_GEMMS
  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // W4A16 grouped skinny GEMM: packed int4 weights, per-group scales,
  // optional zero points for asymmetric quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");
  rocm_ops.impl("wvSplitK_int4_g", torch::kCUDA, &wvSplitK_int4_g);
  rocm_ops.def(
      "moe_skinny_int4_decode(Tensor input, Tensor w13, Tensor w13_scale, "
      "Tensor w2, Tensor w2_scale, Tensor topk_weights, Tensor topk_ids, "
      "Tensor! act_buf, Tensor! output, int group_size, Tensor? expert_map) -> ()");
  rocm_ops.impl("moe_skinny_int4_decode", torch::kCUDA, &moe_skinny_int4_decode);

  // gfx1030 fp16 skinny GEMM for decode-sized M (T43)
  rocm_ops.def("gemv_f16_rdna2(Tensor x, Tensor w, Tensor? bias) -> Tensor");
  rocm_ops.impl("gemv_f16_rdna2", torch::kCUDA, &gemv_f16_rdna2);
  rocm_ops.def("gemv_i8_rdna2(Tensor x, Tensor w, Tensor scale, Tensor? bias) -> Tensor");
  rocm_ops.impl("gemv_i8_rdna2", torch::kCUDA, &gemv_i8_rdna2);

  // T44: push-based one-shot all-reduce for small TP messages on gfx1030
  rocm_ops.def(
      "rdna_ar_init(int rank, int world, Tensor device_ids, int max_bytes, "
      "str shm_name) -> Tensor");
  rocm_ops.impl("rdna_ar_init", torch::kCPU, &rdna_ar_init);
  rocm_ops.def("rdna_ar_connect(int handle, Tensor handles) -> ()");
  rocm_ops.impl("rdna_ar_connect", torch::kCPU, &rdna_ar_connect);
  rocm_ops.def("rdna_ar_can(int handle, Tensor t) -> bool");
  rocm_ops.impl("rdna_ar_can", torch::kCUDA, &rdna_ar_can);
  rocm_ops.def("rdna_ar_all_reduce(int handle, Tensor t) -> Tensor");
  rocm_ops.impl("rdna_ar_all_reduce", torch::kCUDA, &rdna_ar_all_reduce);
  // no tensor arguments -> no dispatch key; register as catch-all
  rocm_ops.def("rdna_ar_timed_out(int handle) -> bool", &rdna_ar_timed_out);
  rocm_ops.def("rdna_ar_fast_calls(int handle) -> int", &rdna_ar_fast_calls);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);
#endif  // VLLM_SKIP_SKINNY_GEMMS

#ifdef VLLM_ROCM_GFX1100
  // W4A16 GPTQ kernels for AMD RDNA3 (gfx1100).
  rocm_ops.def(
      "gptq_gemm_rdna3(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, Tensor b_g_idx, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, Tensor b_g_idx, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3_wmma", torch::kCUDA, &gptq_gemm_rdna3_wmma);

  rocm_ops.def(
      "moe_gptq_gemm_rdna3(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor b_qzeros, Tensor topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna3", torch::kCUDA, &moe_gptq_gemm_rdna3);
#endif

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
