#pragma once
#include "../macro.h"
#include "../model/network.h"
#include "cudamem.cuh"
namespace MGBrain
{
    /// @brief 神经元数据块
    struct NEUBlock
    {
        /// LIF
        int *ids;
        bool *Fired;
        int *Fire_cnt;
        int *Last_fired;
        real *V_m;
        real *I_exc;
        real *I_inh;
        int *Refrac_state;
        real *I_buffer_exc;
        real *I_buffer_inh;
        /// 泊松
        bool *poisson;
        int *type;
        real *rate;
        curandState *state;
    };
    
    /// @brief 突触数据块
    struct SYNBlock
    {
        int *src;
        int *tar;
        real *weight;
        int *delay;
    };
    /// @brief LIF神经元模型常量参数
    struct LIFConsts
    {
        real P22;
        real P11exc;
        real P11inh;
        real P21exc;
        real P21inh;
        real C_m;
        real V_reset;
        real V_rest;
        real V_thresh;
        real Tau_m;
        real Tau_exc;
        real Tau_inh;
        real I_offset;
        real Tau_refrac;
    };
    /// @brief STDP突触模型常量参数
    struct STDPConsts
    {
        real A_LTP;
        real A_LTD;
        real TAU_LTP;
        real TAU_LTD;
        real W_max;
        real W_min;
    };

};
