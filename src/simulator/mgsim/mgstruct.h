#pragma once
#include "../structure.h"
#include <omp.h>
namespace MGBrain{
    /// @brief 外部突触块
    struct OUTSYNBlock
    {
        // 目标网络id
        int tar_id;
        // 突触块大小
        int syn_size;
        // 突触块
        SYNBlock block;
    };
};