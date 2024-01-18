/**
 * 突触分散管理方式，每个神经元的突触分别存放。
*/
#pragma once
#include "../macro.h"
#include "../model/network.h"
#include "cudamem.cuh"
namespace MGBrain
{
    struct DendBlock{
        int size;
        int *src;
    };
    struct AxonBlock{
        int size;
        int* tar;
        real* weight;
        int* delay;
    };
    struct NeuBlock
    {
        /// 基础信息
        int size;
        int *ids;
        bool *fired;
        int* fire_cnt;
        int* last_fired;
        int* type;
        /// 生物学属性
        int* Refrac_state;
        real* V_m;
        real* I_exc;
        real* I_inh;
        real* I_buffer_exc;
        real* I_buffer_inh;
        /// 泊松相关
        
        /// @brief 激活率
        real* rate;
        /// @brief 随机种子，用于CUDA生成随机数
        curandState *state;
        
        /// 突触相关

        /// @brief 轴突
        AxonBlock* axons;
        /// @brief 树突
        DendBlock* dends;
    };
    
};
