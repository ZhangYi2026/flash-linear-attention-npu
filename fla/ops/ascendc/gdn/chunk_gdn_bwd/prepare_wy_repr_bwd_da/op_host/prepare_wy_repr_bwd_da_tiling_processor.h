/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_da_tiling_processor.h
 * \brief Tiling processor shared by aclnn tiling and fast kernel launch.
 */

#ifndef PREPARE_WY_REPR_BWD_DA_TILING_PROCESSOR_H
#define PREPARE_WY_REPR_BWD_DA_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

#include "../op_kernel/prepare_wy_repr_bwd_da_struct.h"

using GDN::PrepareWyReprBwdDaTilingData;

namespace optiling {

static constexpr size_t INPUT_K_IDX = 0;
static constexpr size_t INPUT_V_IDX = 1;
static constexpr size_t INPUT_BETA_IDX = 2;
static constexpr size_t INPUT_A_IDX = 3;
static constexpr size_t INPUT_DW_IDX = 4;
static constexpr size_t INPUT_DU_IDX = 5;
static constexpr size_t INPUT_G_IDX = 6;
static constexpr size_t INPUT_SEQLENS_IDX = 7;
static constexpr size_t INPUT_CHUNK_INDICES_IDX = 8;

static constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;

static constexpr size_t DIM_NUM_3 = 3;
static constexpr size_t DIM_NUM_4 = 4;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;

static constexpr int64_t CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_SIZE_128 = 128;
static constexpr int64_t VAR_LEN_B_DIM_1 = 1;

static constexpr int64_t PREPARE_WY_REPR_BWD_DA_DTYPE_FP16 = 0;
static constexpr int64_t PREPARE_WY_REPR_BWD_DA_DTYPE_BF16 = 1;
static constexpr int64_t PREPARE_WY_REPR_BWD_DA_DTYPE_FP32 = 2;

static constexpr const char *const INPUT_K_NAME = "k";
static constexpr const char *const INPUT_V_NAME = "v";
static constexpr const char *const INPUT_BETA_NAME = "beta";
static constexpr const char *const INPUT_A_NAME = "A";
static constexpr const char *const INPUT_DW_NAME = "dw";
static constexpr const char *const INPUT_DU_NAME = "du";
static constexpr const char *const INPUT_G_NAME = "g";
static constexpr const char *const INPUT_CHUNK_INDICES_NAME = "chunk_indices";
static constexpr const char *const INPUT_SEQLENS_NAME = "cu_seqlens";

static constexpr uint64_t SIZE_HALF = 2;
static constexpr uint64_t SIZE_FP32 = 4;
static constexpr uint64_t ONE_BLOCK_32 = 32;
static constexpr uint64_t BIT_NUM_FOR_UINT8 = 8;

struct PrepareWyReprBwdDaTilingContext {
    const char *nodeName;
    const gert::StorageShape *kShape;
    const gert::StorageShape *vShape;
    const gert::StorageShape *betaShape;
    const gert::StorageShape *aShape;
    const gert::StorageShape *dwShape;
    const gert::StorageShape *duShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    const int64_t *cuSeqlensData;
    const int64_t *chunkIndicesData;
    int64_t chunkSize;
    int64_t kDataType;
    int64_t betaDataType;
    uint64_t ubSize;
    uint32_t aicCoreNum;
    size_t sysWorkspaceSize;
};

class PrepareWyReprBwdDaTilingProcessor {
    PrepareWyReprBwdDaTilingContext &ctx_;
    PrepareWyReprBwdDaTilingData &tiling_;
    size_t workspaceSize_ = 0;
    uint32_t blockDim_ = 0;

public:
    explicit PrepareWyReprBwdDaTilingProcessor(PrepareWyReprBwdDaTilingContext &ctx,
                                               PrepareWyReprBwdDaTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    size_t GetWorkspaceSize() const
    {
        return workspaceSize_;
    }

    uint32_t GetBlockDim() const
    {
        return blockDim_;
    }

    uint32_t GetTilingKey() const
    {
        return 1U;
    }

    bool IsVariableLength() const
    {
        return ctx_.cuSeqlensShape != nullptr || ctx_.chunkIndicesShape != nullptr;
    }

    ge::graphStatus RequiredInputDimNumCheck(const gert::StorageShape *curShape, size_t validDimNum,
                                             const char *inputName)
    {
        OP_CHECK_IF(curShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s is required, but got nullptr.", inputName),
                    return ge::GRAPH_FAILED);
        const gert::Shape storageShape = curShape->GetStorageShape();
        size_t dimNum = storageShape.GetDimNum();
        OP_CHECK_IF(dimNum != validDimNum,
                    OP_LOGE(ctx_.nodeName,
                            "Check input %s shape failed, the dim num should be %zu, but get %zu.", inputName,
                            validDimNum, dimNum),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PreCheck()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.kShape, DIM_NUM_4, INPUT_K_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.vShape, DIM_NUM_4, INPUT_V_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.betaShape, DIM_NUM_3, INPUT_BETA_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.aShape, DIM_NUM_4, INPUT_A_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.dwShape, DIM_NUM_4, INPUT_DW_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.duShape, DIM_NUM_4, INPUT_DU_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.gShape, DIM_NUM_3, INPUT_G_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CompareShape(const gert::Shape &shape1, const gert::Shape &shape2, const char *inputName1,
                                 const char *inputName2, size_t compareDimNum)
    {
        for (size_t dimIndex = 0; dimIndex < compareDimNum; dimIndex++) {
            size_t shapeDim1 = shape1.GetDim(dimIndex);
            size_t shapeDim2 = shape2.GetDim(dimIndex);
            OP_CHECK_IF(shapeDim1 != shapeDim2,
                        OP_LOGE(ctx_.nodeName,
                                "Compare input shape of %s and %s failed, the length of dim %zu should be same,but got "
                                "%zu and %zu.",
                                inputName1, inputName2, dimIndex, shapeDim1, shapeDim2),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    uint64_t DtypeSize(int64_t dtype) const
    {
        return dtype == PREPARE_WY_REPR_BWD_DA_DTYPE_FP32 ? SIZE_FP32 : SIZE_HALF;
    }

    ge::graphStatus SetRowNumKBetaG(uint64_t ubSize)
    {
        uint64_t rowNum = tiling_.chunkSize / 2;
        uint64_t sizeofKType = DtypeSize(ctx_.kDataType);
        uint64_t sizeofBetaType = DtypeSize(ctx_.betaDataType);
        while (rowNum >= 8) {
            uint64_t useUbSize = 0;
            useUbSize += 2 * rowNum * tiling_.K * sizeofKType;
            useUbSize += 2 * rowNum * sizeofBetaType;
            useUbSize += 2 * rowNum * tiling_.K * sizeofBetaType;
            useUbSize += rowNum * tiling_.K * SIZE_FP32;
            useUbSize += rowNum * SIZE_FP32;
            useUbSize += rowNum * SIZE_FP32;
            useUbSize += rowNum * ONE_BLOCK_32;
            useUbSize += 2 * rowNum * tiling_.K * sizeofKType;
            if (useUbSize <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        tiling_.rowNumKBetaG = static_cast<int64_t>(rowNum);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetRowNumVBeta(uint64_t ubSize)
    {
        uint64_t rowNum = tiling_.chunkSize / 2;
        uint64_t sizeofKType = DtypeSize(ctx_.kDataType);
        uint64_t sizeofBetaType = DtypeSize(ctx_.betaDataType);
        while (rowNum >= 8) {
            uint64_t useUbSize = 0;
            useUbSize += 2 * rowNum * tiling_.V * sizeofKType;
            useUbSize += 2 * rowNum * sizeofBetaType;
            useUbSize += rowNum * tiling_.V * SIZE_FP32;
            useUbSize += rowNum * SIZE_FP32;
            useUbSize += rowNum * ONE_BLOCK_32;
            useUbSize += 2 * rowNum * tiling_.V * sizeofKType;
            if (useUbSize <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        tiling_.rowNumVBeta = static_cast<int64_t>(rowNum);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetRowNumMDuDw(uint64_t ubSize)
    {
        uint64_t rowNum = tiling_.chunkSize / 2;
        uint64_t sizeofKType = DtypeSize(ctx_.kDataType);
        while (rowNum >= 8) {
            uint64_t useUbSize = 0;
            useUbSize += 2 * rowNum * tiling_.chunkSize * sizeofKType;
            useUbSize += 2 * rowNum * tiling_.chunkSize * sizeofKType;
            useUbSize += rowNum * tiling_.chunkSize * SIZE_FP32;
            useUbSize += rowNum * tiling_.chunkSize * SIZE_FP32;
            useUbSize += rowNum * tiling_.chunkSize * SIZE_FP32;
            useUbSize += tiling_.chunkSize * tiling_.chunkSize / BIT_NUM_FOR_UINT8;
            useUbSize += ONE_BLOCK_32;
            useUbSize += 2 * rowNum * tiling_.chunkSize * sizeofKType;
            if (useUbSize <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        tiling_.rowNumMDuDw = static_cast<int64_t>(rowNum);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetRowNumG(uint64_t ubSize)
    {
        uint64_t rowNum = tiling_.chunkSize / 2;
        uint64_t sizeofKType = DtypeSize(ctx_.kDataType);
        uint64_t sizeofBetaType = DtypeSize(ctx_.betaDataType);
        while (rowNum >= 8) {
            uint64_t useUbSize = 0;
            useUbSize += 2 * rowNum * sizeofBetaType;
            useUbSize += 2 * tiling_.chunkSize * sizeofBetaType;
            useUbSize += 2 * rowNum * tiling_.chunkSize * sizeofKType;
            useUbSize += rowNum * SIZE_FP32;
            useUbSize += tiling_.chunkSize * SIZE_FP32;
            useUbSize += rowNum * tiling_.chunkSize * SIZE_FP32;
            useUbSize += rowNum * ONE_BLOCK_32;
            useUbSize += rowNum * tiling_.chunkSize * SIZE_FP32;
            useUbSize += tiling_.chunkSize * tiling_.chunkSize / BIT_NUM_FOR_UINT8;
            useUbSize += ONE_BLOCK_32;
            useUbSize += 2 * rowNum * tiling_.chunkSize * sizeofKType;
            if (useUbSize <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        tiling_.rowNumG = static_cast<int64_t>(rowNum);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CommonTiling()
    {
        const gert::Shape kStorageShape = ctx_.kShape->GetStorageShape();
        const gert::Shape vStorageShape = ctx_.vShape->GetStorageShape();
        const gert::Shape betaStorageShape = ctx_.betaShape->GetStorageShape();
        const gert::Shape aStorageShape = ctx_.aShape->GetStorageShape();
        const gert::Shape dwStorageShape = ctx_.dwShape->GetStorageShape();
        const gert::Shape duStorageShape = ctx_.duShape->GetStorageShape();
        const gert::Shape gStorageShape = ctx_.gShape->GetStorageShape();

        OP_CHECK_IF(CompareShape(vStorageShape, duStorageShape, INPUT_V_NAME, INPUT_DU_NAME, DIM_NUM_4) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(betaStorageShape, gStorageShape, INPUT_BETA_NAME, INPUT_G_NAME, DIM_NUM_3) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(vStorageShape.GetDim(DIM_0) != kStorageShape.GetDim(DIM_0) ||
                        vStorageShape.GetDim(DIM_2) != kStorageShape.GetDim(DIM_2),
                    OP_LOGE(ctx_.nodeName, "v and k must share batch and time dimensions."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(DIM_0) != gStorageShape.GetDim(DIM_0) ||
                        kStorageShape.GetDim(DIM_2) != gStorageShape.GetDim(DIM_2),
                    OP_LOGE(ctx_.nodeName, "k and g must share batch and time dimensions."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dwStorageShape.GetDim(DIM_0) != kStorageShape.GetDim(DIM_0) ||
                        dwStorageShape.GetDim(DIM_2) != kStorageShape.GetDim(DIM_2) ||
                        dwStorageShape.GetDim(DIM_3) != kStorageShape.GetDim(DIM_3),
                    OP_LOGE(ctx_.nodeName, "dw and k must share batch, time, and K dimensions."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(DIM_0) != aStorageShape.GetDim(DIM_0) ||
                        kStorageShape.GetDim(DIM_2) != aStorageShape.GetDim(DIM_2),
                    OP_LOGE(ctx_.nodeName, "k and A must share batch and time dimensions."),
                    return ge::GRAPH_FAILED);

        tiling_.B = static_cast<int64_t>(vStorageShape.GetDim(DIM_0));
        tiling_.HV = static_cast<int64_t>(vStorageShape.GetDim(DIM_1));
        tiling_.HK = static_cast<int64_t>(kStorageShape.GetDim(DIM_1));
        tiling_.T = static_cast<int64_t>(vStorageShape.GetDim(DIM_2));
        tiling_.K = static_cast<int64_t>(kStorageShape.GetDim(DIM_3));
        tiling_.V = static_cast<int64_t>(vStorageShape.GetDim(DIM_3));

        OP_CHECK_IF(tiling_.HK <= 0 || tiling_.HV <= 0 || tiling_.HV % tiling_.HK != 0,
                    OP_LOGE(ctx_.nodeName,
                            "HV (%ld) must be a positive multiple of HK (%ld) for GVA.", tiling_.HV, tiling_.HK),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.chunkSize != CHUNK_SIZE_64 && ctx_.chunkSize != CHUNK_SIZE_128,
                    OP_LOGE(ctx_.nodeName,
                            "Check attr chunkSize failed, the chunkSize should be 64 or 128, but get %ld.",
                            ctx_.chunkSize),
                    return ge::GRAPH_FAILED);
        tiling_.chunkSize = ctx_.chunkSize;

        OP_CHECK_IF(SetRowNumKBetaG(ctx_.ubSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(ctx_.nodeName, "SetRowNumKBetaG Failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(SetRowNumVBeta(ctx_.ubSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(ctx_.nodeName, "SetRowNumVBeta Failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(SetRowNumMDuDw(ctx_.ubSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(ctx_.nodeName, "SetRowNumMDuDw Failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(SetRowNumG(ctx_.ubSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(ctx_.nodeName, "SetRowNumG Failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    int64_t CeilDiv(int64_t a, int64_t b) const
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    ge::graphStatus FixLenTiling()
    {
        tiling_.chunkNum = tiling_.B * CeilDiv(tiling_.T, tiling_.chunkSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.chunkIndicesShape, DIM_1, INPUT_CHUNK_INDICES_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.cuSeqlensShape, DIM_1, INPUT_SEQLENS_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.cuSeqlensData == nullptr || ctx_.chunkIndicesData == nullptr,
                    OP_LOGE(ctx_.nodeName, "Variable-length tiling requires value data for cu_seqlens and chunk_indices."),
                    return ge::GRAPH_FAILED);

        const gert::Shape seqlensStorageShape = ctx_.cuSeqlensShape->GetStorageShape();
        int64_t seqlensDim0 = seqlensStorageShape.GetDim(DIM_0);
        OP_CHECK_IF(seqlensDim0 < 2,
                    OP_LOGE(ctx_.nodeName,
                            "Check seqlens shape failed, the dim 0 of seqlens should be larger than 1, but get %ld.",
                            seqlensDim0),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.cuSeqlensData[0] != 0,
                    OP_LOGE(ctx_.nodeName, "Check seqlens data failed, the seqlens[0] should be 0, but get %ld.",
                            ctx_.cuSeqlensData[0]),
                    return ge::GRAPH_FAILED);

        std::vector<int64_t> expectChunkIndices;
        for (int64_t i = 1; i < seqlensDim0; i++) {
            int64_t curSeqLen = ctx_.cuSeqlensData[i] - ctx_.cuSeqlensData[i - 1];
            OP_CHECK_IF(curSeqLen <= 0,
                        OP_LOGE(ctx_.nodeName, "Check seqlens data failed, seqlens should be strictly increasing."),
                        return ge::GRAPH_FAILED);
            for (int64_t j = 0; j < curSeqLen; j += tiling_.chunkSize) {
                expectChunkIndices.push_back(i - 1);
                expectChunkIndices.push_back(j / tiling_.chunkSize);
            }
        }

        const gert::Shape chunkIndicesStorageShape = ctx_.chunkIndicesShape->GetStorageShape();
        int64_t chunkIndicesDim0 = chunkIndicesStorageShape.GetDim(DIM_0);
        OP_CHECK_IF(chunkIndicesDim0 != static_cast<int64_t>(expectChunkIndices.size()),
                    OP_LOGE(ctx_.nodeName,
                            "Check chunk_indices shape failed, the len of chunk_indices should be %zu, but get %ld.",
                            expectChunkIndices.size(), chunkIndicesDim0),
                    return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < static_cast<int64_t>(expectChunkIndices.size()); i++) {
            OP_CHECK_IF(expectChunkIndices[i] != ctx_.chunkIndicesData[i],
                        OP_LOGE(ctx_.nodeName,
                                "Check chunk_indices data failed, the chunk_indices[%ld] should be %ld, but get %ld.",
                                i, expectChunkIndices[i], ctx_.chunkIndicesData[i]),
                        return ge::GRAPH_FAILED);
        }
        tiling_.chunkNum = chunkIndicesDim0 / 2;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus WorkspaceTiling()
    {
        blockDim_ = ctx_.aicCoreNum;
        int64_t maxKV = std::max(tiling_.K, tiling_.V);
        size_t userWorkspaceSize = static_cast<size_t>(DtypeSize(ctx_.kDataType)) *
                                   static_cast<size_t>(tiling_.B) * static_cast<size_t>(tiling_.HV) *
                                   static_cast<size_t>(tiling_.T) *
                                   static_cast<size_t>(tiling_.chunkSize + maxKV);
        workspaceSize_ = ctx_.sysWorkspaceSize + userWorkspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CommonTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        if (IsVariableLength()) {
            OP_CHECK_IF(ctx_.cuSeqlensShape == nullptr || ctx_.chunkIndicesShape == nullptr,
                        OP_LOGE(ctx_.nodeName,
                                "Variable-length tiling requires both cu_seqlens and chunk_indices to be provided."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(tiling_.B != VAR_LEN_B_DIM_1,
                        OP_LOGE(ctx_.nodeName,
                                "If cu_seqlens is not nullptr, the dim 0 of q needs to be 1, but now is %ld.",
                                tiling_.B),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(VariableLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVariable = 1;
        } else {
            OP_CHECK_IF(FixLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVariable = 0;
        }
        OP_CHECK_IF(WorkspaceTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace optiling

#endif // PREPARE_WY_REPR_BWD_DA_TILING_PROCESSOR_H
