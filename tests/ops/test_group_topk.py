import pytest
import torch
import torch_npu
from typing import Optional, Tuple, Union
import vllm_ascend.platform


DTYPES = [torch.float16]
# SHAPES = [(3, 4, 3)]
DEVICES = [f"npu:{0}"]
SEEDS = [0]


# from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe.py#L307-L329
def native_grouped_topk(
    topk_weights: torch.Tensor,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
):
    topk_group = 0 if topk_group is None else topk_group
    num_expert_group = 0 if num_expert_group is None else num_expert_group

    num_token = topk_weights.shape[0]
    grouped_weights = (
        topk_weights.view(num_token, num_expert_group, -1).max(dim=-1).values
    )
    topk_group_indices = torch.topk(
        grouped_weights.to(torch.float32), k=topk_group, dim=-1, sorted=False
    )[1]
    topk_group_mask = torch.zeros_like(grouped_weights)
    topk_group_mask.scatter_(1, topk_group_indices, 1)
    topk_weight_mask = (
        topk_group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, topk_weights.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )
    topk_weights = topk_weights.masked_fill(~topk_weight_mask.bool(), 0.0)
    return topk_weights


# @pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_get_masked_input_and_mask(
    # shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    groupNum = 8
    k = 1
    kInner = 1
    tokenNum = 3
    expertNums = 256
    topKInput = torch.rand(tokenNum, expertNums, device=device, dtype=dtype).npu()
    idx_arr =  torch.arange(0, 1024, device=device, dtype=torch.int32).npu()
    input = torch.ops._C.group_topk(
        topKInput,
        idx_arr,
        groupNum,
        k,
        kInner
    )
    golden = native_grouped_topk(topKInput, groupNum, k)
    torch.testing.assert_eq(input, golden)


