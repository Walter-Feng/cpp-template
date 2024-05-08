#ifndef CPP_TEMPLATE_UTIL_NCCL_H
#define CPP_TEMPLATE_UTIL_NCCL_H

#include <nccl.h>

namespace util::nccl {
ncclResult_t ncclCommInitAll(ncclComm_t * comm, int ndev);
}

#endif //CPP_TEMPLATE_UTIL_NCCL_H
