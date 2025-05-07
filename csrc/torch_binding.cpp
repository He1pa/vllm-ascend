/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
#include <pybind11/pybind11.h>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclnn/opdev/platform.h"
#include "ops.h"
#include "utils.h"
#include "kernels/types.h"

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> rotary_embedding(at::Tensor &positions, at::Tensor &query, at::Tensor &key,
    int64_t head_size, at::Tensor &cos_sin_cache,  bool is_neox)
{
    int32_t deviceId = 0;
    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
    TORCH_CHECK(
        positions_ndim == 1 || positions_ndim == 2,
        "positions must have shape [num_tokens] or [batch_size, seq_len]");
    if (positions_ndim == 1) {
      TORCH_CHECK(
          query.size(0) == positions.size(0) && key.size(0) == positions.size(0),
          "query, key and positions must have the same number of tokens");
    }
    if (positions_ndim == 2) {
      TORCH_CHECK(
          query.size(0) == positions.size(0) &&
              key.size(0) == positions.size(0) &&
              query.size(1) == positions.size(1) &&
              key.size(1) == positions.size(1),
          "query, key and positions must have the same batch_size and seq_len");
    }
    TORCH_CHECK(head_size % 32 == 0, "rotary_embedding: headSize should be divisible by 32");
    int query_hidden_size = query.numel() / num_tokens;
    int key_hidden_size = key.numel() / num_tokens;
    TORCH_CHECK(query_hidden_size % head_size == 0);
    TORCH_CHECK(key_hidden_size % head_size == 0);
    TORCH_CHECK(is_neox == true, "rotary_embedding: neox=false is not supported as custom kernel in vllm-ascend");

    // Make sure query and key have consistent number of heads
    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = key_hidden_size / head_size;
    TORCH_CHECK(num_heads % num_kv_heads == 0);
    at::Tensor query_dst = at::empty({num_tokens, num_heads, head_size}, query.options());
    at::Tensor key_dst = at::empty({num_tokens, num_kv_heads, head_size}, key.options());

    int rot_dim = cos_sin_cache.size(1);
    int seq_dim_idx = positions_ndim - 1;
    int64_t *position_ids_ptr = positions.data_ptr<int64_t>();
    void *query_dst_ptr = query_dst.data_ptr();
    void *key_dst_ptr = key_dst.data_ptr();
    void *query_ptr = query.data_ptr();
    void *key_ptr = key.data_ptr();
    void *cos_sin_cache_ptr = cos_sin_cache.data_ptr();
    int64_t query_stride = query.stride(seq_dim_idx);
    int64_t key_stride = key.stride(seq_dim_idx);
    int64_t dst_query_stride = query_dst.stride(0);
    int64_t dst_key_stride = key_dst.stride(0);
    at::ScalarType scalar_type = query.scalar_type();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("rotary_embedding");
    cmd.SetCustomHandler([scalar_type, is_neox, num_tokens, stream, position_ids_ptr, query_dst_ptr, key_dst_ptr,
                          query_ptr, key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride,
                          dst_query_stride, dst_key_stride, num_heads, num_kv_heads, head_size]() -> int {
        auto dtype_num = get_dtype_from_torch(scalar_type);
        fe::PlatFormInfos platform_infos;
        int device_id = 0;
        fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos);
        uint32_t aivNum = platform_infos.GetCoreNumByType("aiv");
        uint32_t loop_cnt = (num_tokens + aivNum - 1) / aivNum;
        rotary_embedding_impl(dtype_num, is_neox, stream, position_ids_ptr, query_dst_ptr, key_dst_ptr, query_ptr,
                                key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride, dst_query_stride,
                                dst_key_stride, num_heads, num_kv_heads, head_size, num_tokens, loop_cnt, aivNum);
        return 0;
    });
    cmd.Run();
    return {query_dst, key_dst};
}

constexpr uint32_t DATABLOCK_SIZE = 32;
constexpr uint32_t SORT_REPEAT_COUNT = 32;
constexpr uint32_t B16_SIZE = 2;
constexpr uint32_t B32_SIZE = 4;
constexpr uint32_t DATABLOCK_NUM_PER_REPEAT = 8;
constexpr uint32_t B16_PER_BLOCK = DATABLOCK_SIZE / B16_SIZE;
constexpr uint32_t MAX_TILE_SIZE = 16 * 1024;
constexpr uint64_t SINGLE_VALUE_GROUP_TILING_KEY = 900;
constexpr uint64_t TILING_KEY_COEFFICIENT = 10;
constexpr uint64_t TILING_KEY_VERSION = 20000000000;

at::Tensor group_topk(at::Tensor &topKInput, at::Tensor &idxArr, const uint32_t groupNum, const uint32_t k, const uint32_t kInner)
{
    TORCH_CHECK(
        topKInput.dim() == 2,
        "inputs must 2 dim");

    const uint32_t tokenNum = topKInput.size(0)
    const uint32_t expertNum = topKInput.size(1)

    at::ScalarType scalar_type = topKInput.scalar_type();
    auto dtype_num = get_dtype_from_torch(scalar_type);
    const uint32_t expertNumPerToken = expertNum;
    const uint32_t expertNumPerGroup = expertNum / groupNum;
    const uint32_t expertNumPerGroupAlign = kInner > 1 ? SORT_REPEAT_COUNT : B16_PER_BLOCK;
    const uint32_t expertNumPerGroupPadded = (expertNumPerGroup + expertNumPerGroupAlign - 1) / expertNumPerGroupAlign * expertNumPerGroupAlign; // RoundUP

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    uint64_t tilingKey = 0;
    if (expertNumPerGroup != 1)
    {
        tilingKey = static_cast<uint64_t>(kInner > 1);
        tilingKey = tilingKey * TILING_KEY_COEFFICIENT + static_cast<uint64_t>(dtype_num == AscendType::BF16);
        tilingKey += TILING_KEY_VERSION;
    }
    else
    {
        tilingKey = SINGLE_VALUE_GROUP_TILING_KEY + static_cast<uint64_t>(dtype_num == AscendType::BF16);
    }

    // Create and configure OpCommand
    at_npu::native::OpCommand cmd;
    cmd.SetCustomHandler([topKInput, idxArr, groupNum, k, kInner, expertNumPerGroup,
                            expertNumPerGroupPadded, expertNumPerToken,, tilingKey, stream]() -> int
                            {
        fe::PlatFormInfos platform_infos;
        int device_id = 0;
        fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platform_infos);
        auto coreNum = ascendcPlatform.GetCoreNum();
        
        uint32_t aivNum = platform_infos.GetCoreNumByType("aiv");

        const uint32_t tokenNumPerCore = tokenNum / coreNum;
        const uint32_t tailTokenNum = tokenNum % coreNum;

        group_topk(topKInput, idxArr, groupNum, k, kInner, expertNumPerGroup,
            expertNumPerGroupPadded, expertNumPerToken, tokenNumPerCore, tailTokenNum, tilingKey, stream, aivNum);
        return 0;
    });
    cmd.Run();
    return topKInput
}
} // namespace vllm_ascend

TORCH_LIBRARY_EXPAND(_C, ops)
{
    // vLLM-Ascend custom ops
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", torch::kPrivateUse1, &vllm_ascend::weak_ref_tensor);

    // Rotary embedding
    // Apply GPT-NeoX style rotary embedding to query and key.
    ops.def(
        "rotary_embedding(Tensor positions, Tensor! query,"
        "                 Tensor! key, int head_size,"
        "                 Tensor cos_sin_cache, bool is_neox) -> (Tensor query, Tensor key)");
    ops.impl("rotary_embedding", torch::kPrivateUse1, &vllm_ascend::rotary_embedding);

    ops.def(
        "group_topk(Tensor topKInput, "
        "           Tensor idxArr, "
        "           int groupNum, "
        "           int k, "
        "           int kInner) -> (Tensor masked_input, Tensor mask)");
    ops.impl("group_topk", torch::kPrivateUse1, &vllm_ascend::group_topk);
}

REGISTER_EXTENSION(_C)
