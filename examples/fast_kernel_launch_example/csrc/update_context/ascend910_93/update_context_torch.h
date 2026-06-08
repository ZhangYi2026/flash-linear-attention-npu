#ifndef UPDATE_CONTEXT_TORCH_H
#define UPDATE_CONTEXT_TORCH_H
#include <torch/extension.h>
#include "hccl/hccl.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_mc2.h"
#include "hccl/hccl_res_expt.h"

struct Mc2ContextStru {
    uint64_t epRankId;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer[1024];
};


std::tuple<at::Tensor, int64_t> update_context(const at::Tensor &x, c10::string_view group_ep, int64_t ep_world_size);


#endif
