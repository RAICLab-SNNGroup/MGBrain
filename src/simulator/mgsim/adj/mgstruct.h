#pragma once
#include "../../ustruct.h"
#include "spikebuffer.h"
#include "spikebufferd.h"
#include <omp.h>
namespace MGBrain{
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
    /// @brief 邻接信息数据块
    struct ADJBlock
    {
        size_t *axon_offs;
        size_t *axon_refs;
        size_t *dend_offs;
        size_t *dend_refs;
    };
    /// @brief 多区域子网络
    struct GSubNet
    {
        /// @brief 子网络号
        int id;
        /// @brief 网络数量
        int npart;
        /// @brief 神经元数据块
        NEUBlock neus;
        /// @brief 突触数据块
        SYNBlock syns;
        /// @brief 邻接信息块
        ADJBlock adjs;
        
        size_t* out_syn_size_list;
        int*    out_net_id_list;
        
    };
    
    
    /// @brief 用于仿真STDP模型时，所需跨设备地址信息
    struct GNetAddrs{
        int **glast_fired_addrs;
        int **gsyn_src_addrs;
        real **gsyn_weight_addrs;
    };
    
    
};