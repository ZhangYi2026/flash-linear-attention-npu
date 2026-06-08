
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CHUNK_BWD_DV_LOCAL_H_
#define ACLNN_CHUNK_BWD_DV_LOCAL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnChunkBwdDvLocalGetWorkspaceSize
 * parameters :
 * q : required
 * k : required
 * dO : required
 * g : required
 * upperTriMatrixOptional : optional
 * gGammaOptional : optional
 * aOptional : optional
 * cuSeqlensOptional : optional
 * chunkIndicesOptional : optional
 * scale : required
 * chunkSize : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkBwdDvLocalGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *dO,
    const aclTensor *g,
    const aclTensor *gGammaOptional,
    const aclTensor *aOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnChunkBwdDvLocal
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkBwdDvLocal(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
