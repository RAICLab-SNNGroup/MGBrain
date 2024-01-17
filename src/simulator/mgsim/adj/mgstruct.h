#pragma once
#include "../../ustruct.h"
#include <omp.h>
namespace MGBrain{
    /// @brief 邻接信息数据块
    struct ADJBlock
    {
        size_t *axon_offs;
        size_t *axon_refs;
        size_t *dend_offs;
        size_t *dend_refs;
    };
};