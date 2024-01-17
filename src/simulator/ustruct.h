/**
 * 突触集中管理方式，每个神经元的突触集中存放。
*/
#pragma once
#include "../macro.h"
#include "../model/network.h"
#include "cudamem.cuh"
namespace MGBrain
{
    /// @brief 神经元数据块
    struct NEUBlock
    {
        
        size_t size;
        /// 基本信息
        int *ids;
        int *type;
        bool *source;
        bool *fired;
        int *fire_cnt;
        int *last_fired;
        /// 生物学信息
        int *Refrac_state;
        real *V_m;
        real *I_exc;
        real *I_inh;
        real *I_buffer_exc;
        real *I_buffer_inh;
        /// 泊松
        
        real *rate;
        curandState *state;
    };
    /// @brief 突触数据块
    struct SYNBlock
    {
        size_t size;
        int *src;
        int *tar;
        real *weight;
        int *delay;
    };
    
    
    
    

};
