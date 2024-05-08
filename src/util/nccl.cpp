#include "nccl.h"

namespace util::nccl {
ncclResult_t ncclCommInitAll(ncclComm_t * comm, int ndev) {
  ncclUniqueId Id;
  ncclGetUniqueId(&Id);
  ncclGroupStart();
  for (int i = 0; i < ndev; i++) {
    cudaSetDevice(i);
    ncclCommInitRank(comm + i, ndev, Id, i);
  }
  return ncclGroupEnd();
}
}