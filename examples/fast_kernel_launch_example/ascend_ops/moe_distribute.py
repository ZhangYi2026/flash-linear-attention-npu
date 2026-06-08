import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error

X_CONTEXT_SIZE=4

class IntWrapper:
    def __init__(self, value=0):
        self.value = value

class MoeDistributeBuffer:
    def __init__(self, group, comm_alg: int = 0):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.world_size = torch.distributed.get_world_size(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        x_context = torch.zeros(X_CONTEXT_SIZE, dtype=torch.int32).npu()
        self.context, self.ccl_buffer_size = torch.ops.ascend_ops.updateContext(x_context, self.group_name, self.world_size)

    @staticmethod
    def get_ccl_buffer_size(world_size: int, num_max_dispatch_tokens_per_rank: int, hidden: int,
                                        num_moe_expert: int, topk: int, num_shared_expert: int = 0,
                                        num_shared_expert_ranks: int = 0, comm_alg: str = "",
                                        ) -> int:
        def inline_align(value, base):
            return (value + base - 1) // base * base

        max_out_dtype_size = 2 # sizeof(int32)
        mb_conversion = 1024 * 1024
        ub_align = 32 # 32B
        scale_expand_index_buffer = 44 # scale 32B + 3 * 4 expand_idx
        full_mesh_data_align = 480
        win_addr_align = 512

        comm_alg_support_list = ["fullmesh_v1", "fullmesh_v2", ""]
        torch._check(comm_alg in comm_alg_support_list,
                     lambda: (f"comm_alg only support {comm_alg_support_list=} "
                              f"but got {comm_alg=}."),)

        token_actual_len = inline_align(hidden * max_out_dtype_size, ub_align) + scale_expand_index_buffer
        if (comm_alg == "fullmesh_v2"):
            token_need_size_dispatch = inline_align(token_actual_len, full_mesh_data_align) // full_mesh_data_align * \
                win_addr_align
        else: 
            token_need_size_dispatch = inline_align(token_actual_len, win_addr_align)
        token_need_size_combine = inline_align(hidden * max_out_dtype_size, win_addr_align)

        local_moe_expert_num = num_moe_expert // (world_size - num_shared_expert_ranks)
        minimum_buffer_size = 2 * (
            (num_max_dispatch_tokens_per_rank * token_need_size_dispatch * world_size * local_moe_expert_num) + \
            (num_max_dispatch_tokens_per_rank * token_need_size_combine * (topk + num_shared_expert))) + mb_conversion
        return inline_align(inline_align(minimum_buffer_size, mb_conversion) // mb_conversion, 2) // 2

    def update_ctx(self, new_group):
        self.group = new_group
        self.rank_id = torch.distributed.get_rank(new_group)
        self.group_name = new_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        x_context = torch.zeros(X_CONTEXT_SIZE, dtype=torch.int32).npu()
        self.context, self.ccl_buffer_size = torch.ops.ascend_ops.updateContext(x_context, self.group_name, self.world_size)
        return

    def npu_moe_distribute_dispatch_v2(self, x, expert_ids,
            moe_expert_num, *, scales=None, x_active_mask=None,
            expert_scales=None, performance_info=None, expert_shard_type=0, shared_expert_num=0,
            shared_expert_rank_num=0, quant_mode=0, global_bs=0, expert_token_nums_type=1,
            comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0):
        (expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, expand_scales) \
            = torch.ops.ascend_ops.MoeDistributeDispatchV2(
                                             mc2_context=self.context,
                                             x=x,
                                             expert_ids=expert_ids,
                                             group_ep=self.group_name,
                                             ep_world_size=self.world_size,
                                             ep_rank_id=self.rank_id,
                                             moe_expert_num=moe_expert_num,
                                             total_winsize_ep=self.ccl_buffer_size,
                                             scales=scales,
                                             x_active_mask=x_active_mask,
                                             expert_scales=expert_scales,
                                             performance_info=performance_info,
                                             expert_shard_type=expert_shard_type,
                                             shared_expert_num=shared_expert_num,
                                             shared_expert_rank_num=shared_expert_rank_num,
                                             quant_mode=quant_mode,
                                             expert_token_nums_type=expert_token_nums_type,
                                             global_bs=global_bs,
                                             comm_alg=comm_alg,
                                             zero_expert_num=zero_expert_num,
                                             copy_expert_num=copy_expert_num,
                                             const_expert_num=const_expert_num)
        return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, expand_scales


    def npu_moe_distribute_combine_v2(self, expand_x, expert_ids, assist_info_for_combine,
            ep_send_counts, expert_scales,
            moe_expert_num, *,
            x_active_mask=None, shared_expert_x=None, ori_x=None,
            const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None,
            performance_info=None, expert_shard_type=0, shared_expert_num=0, shared_expert_rank_num=0,
            global_bs=0, comm_quant_mode=0,
            comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0):
        return torch.ops.ascend_ops.MoeDistributeCombineV2(
                                             mc2_context=self.context,
                                             expand_x=expand_x,
                                             expert_ids=expert_ids,
                                             assist_info_for_combine=assist_info_for_combine,
                                             ep_send_counts=ep_send_counts,
                                             expert_scales=expert_scales,
                                             group_ep=self.group_name,
                                             ep_world_size=self.world_size,
                                             ep_rank_id=self.rank_id,
                                             moe_expert_num=moe_expert_num,
                                             total_winsize_ep=self.ccl_buffer_size,
                                             x_active_mask=x_active_mask,
                                             shared_expert_x=shared_expert_x,
                                             ori_x=ori_x,
                                             const_expert_alpha_1=const_expert_alpha_1,
                                             const_expert_alpha_2=const_expert_alpha_2,
                                             const_expert_v=const_expert_v,
                                             performance_info=performance_info,
                                             expert_shard_type=expert_shard_type,
                                             shared_expert_num=shared_expert_num,
                                             shared_expert_rank_num=shared_expert_rank_num,
                                             global_bs=global_bs,
                                             comm_quant_mode=comm_quant_mode,
                                             comm_alg=comm_alg,
                                             zero_expert_num=zero_expert_num,
                                             copy_expert_num=copy_expert_num,
                                             const_expert_num=const_expert_num)
