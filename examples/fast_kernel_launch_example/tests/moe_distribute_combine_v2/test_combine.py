import os
import torch
import random
import numpy as np
from torch.multiprocessing import Process
import torch_npu
import torch.distributed as dist
import ascend_ops
from ascend_ops import MoeDistributeBuffer

HCCL_BUFFER_SIZE = 12288


class Config:
    """Configuration for MoE operator test."""
    def __init__(self):
        self.quant_mode = 2
        self.is_dispatch_scales = True
        self.input_dtype = torch.bfloat16
        self.server_num = 1
        self.server_index = 0
        self.port = 50001
        self.master_ip = "[::1]"
        self.dev_num = 16
        self.shared_expert_rank_num = 0
        self.shared_expert_num = 0
        self.num_experts = 32
        self.batch_size = 8
        self.hidden_size = 7168
        self.topk = 8
        self.comm_alg = "fullmesh_v1"
        self.zero_expert_num = 1
        self.copy_expert_num = 1
        self.const_expert_num = 0
        self.random_seed = 0
        
        self.world_size = self.server_num * self.dev_num
        self.ep_world_size = self.world_size
        self.moe_rank_num = self.ep_world_size - self.shared_expert_rank_num
        self.local_num_experts = self.num_experts // self.moe_rank_num
        self.global_batch_size = self.batch_size * self.ep_world_size
        self.is_quant = self.quant_mode > 0


cfg = Config()

buff_size = MoeDistributeBuffer.get_ccl_buffer_size(cfg.world_size, cfg.batch_size, cfg.hidden_size, cfg.num_experts,
                                                    cfg.topk, cfg.shared_expert_num, cfg.shared_expert_rank_num,
                                                    cfg.comm_alg)
buff_size_str = f'{buff_size}'
os.environ['HCCL_BUFFSIZE'] = buff_size_str

def gen_const_tensor(shape):
    """Generate constant expert tensor."""
    if cfg.const_expert_num == 0:
        return None
    return torch.empty(shape, dtype=cfg.input_dtype).uniform_(-1, 1)


def create_process_groups(rank):
    """Create EP process groups."""
    ep_group = None
    ep_ranks = list(range(cfg.ep_world_size))
    ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
    return ep_group


def get_hccl_name(rank, group):
    """Get HCCL communication name."""
    if torch.__version__ > '2.0.1':
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def init_dist(rank):
    """Initialize distributed environment."""
    global_rank = rank + cfg.dev_num * cfg.server_index
    dist.init_process_group(
        backend='hccl', rank=global_rank, world_size=cfg.world_size,
        init_method=f"tcp://{cfg.master_ip}:{cfg.port}",
    )
    ep_group = create_process_groups(global_rank)
    return get_hccl_name(global_rank, ep_group), global_rank, ep_group


def run_npu_process(rank):
    """Run MoE operator test on NPU."""
    torch_npu.npu.set_device(rank)
    ep_hcomm, global_rank, ep_group = init_dist(rank)
    
    x = torch.randn(cfg.batch_size, cfg.hidden_size, dtype=cfg.input_dtype).npu()
    topk_idx = torch.tensor([
        [5, 7, 17, 4, 2, 6, 11, 16], [10, 12, 13, 15, 19, 4, 18, 1],
        [19, 33, 1, 17, 9, 5, 0, 32], [19, 11, 17, 0, 10, 5, 7, 9],
        [10, 16, 11, 17, 33, 8, 9, 3], [12, 19, 5, 7, 1, 3, 18, 16],
        [11, 9, 13, 16, 12, 33, 17, 14], [16, 4, 9, 5, 0, 10, 11, 17],
    ], dtype=torch.int32).npu()
    expert_scales = torch.randn(cfg.batch_size, cfg.topk, dtype=torch.float32).npu()
    
    scales = None
    if cfg.is_dispatch_scales:
        shape = (1 + cfg.num_experts, cfg.hidden_size) if cfg.shared_expert_rank_num else (cfg.num_experts, cfg.hidden_size)
        scales = torch.randn(shape, dtype=torch.float32).npu()
    
    const_alpha_1 = gen_const_tensor([cfg.const_expert_num])
    const_alpha_2 = gen_const_tensor([cfg.const_expert_num])
    const_v = gen_const_tensor([cfg.const_expert_num, cfg.hidden_size])
    
    distribute_buffer = MoeDistributeBuffer(ep_group, buff_size)
    dispatch_fn = distribute_buffer.npu_moe_distribute_dispatch_v2
    expand_x, _, assist_info, _, ep_recv_counts, _ = dispatch_fn(
        x=x, expert_ids=topk_idx, comm_alg=cfg.comm_alg,
        shared_expert_num=cfg.shared_expert_num, shared_expert_rank_num=cfg.shared_expert_rank_num,
        moe_expert_num=cfg.num_experts, quant_mode=cfg.quant_mode,
        global_bs=cfg.global_batch_size, zero_expert_num=cfg.zero_expert_num,
        copy_expert_num=cfg.copy_expert_num, const_expert_num=cfg.const_expert_num,
    )
    
    if cfg.is_quant:
        expand_x = expand_x.to(cfg.input_dtype)
    
    combine_fn = distribute_buffer.npu_moe_distribute_combine_v2
    _ = combine_fn(
        expand_x=expand_x, expert_ids=topk_idx,
        moe_expert_num=cfg.num_experts, assist_info_for_combine=assist_info,
        ep_send_counts=ep_recv_counts, expert_scales=expert_scales,
        shared_expert_num=cfg.shared_expert_num, shared_expert_rank_num=cfg.shared_expert_rank_num,
        global_bs=cfg.global_batch_size, ori_x=x,
        const_expert_alpha_1=const_alpha_1, const_expert_alpha_2=const_alpha_2,
        const_expert_v=const_v, zero_expert_num=cfg.zero_expert_num,
        copy_expert_num=cfg.copy_expert_num, const_expert_num=cfg.const_expert_num,
    )
    
    print(f"rank {global_rank} epid {global_rank}  npu finished!\n")


def validate_config():
    """Validate configuration parameters."""
    if cfg.shared_expert_rank_num > cfg.ep_world_size:
        print("shared_expert_rank_num > ep_world_size"); return False
    if cfg.shared_expert_rank_num > 0 and cfg.ep_world_size % cfg.shared_expert_rank_num != 0:
        print("ep_world_size must be multiple of shared_expert_rank_num"); return False
    if cfg.num_experts % cfg.moe_rank_num != 0:
        print("num_experts must be multiple of moe_rank_num"); return False
    return True


if __name__ == "__main__":
    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    
    print(f"bs={cfg.batch_size}, num_experts={cfg.num_experts}, k={cfg.topk}")
    print(f"quant_mode={cfg.quant_mode}, ep={cfg.ep_world_size}")
    
    if not validate_config():
        exit(0)
    
    procs = []
    for rank in range(cfg.dev_num):
        p = Process(target=run_npu_process, args=(rank,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    
    print("Run NPU success.")
