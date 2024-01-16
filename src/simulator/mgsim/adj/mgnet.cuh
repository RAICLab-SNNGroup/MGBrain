#pragma once
// #include "../mgstruct.h"
#include "spikeBuffer.cuh"
#include "spikeDenseBuffer.cuh"
namespace MGBrain
{
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
        /// @brief 神经元数量
        int neus_size;
        /// @brief 神经元数据块
        NEUBlock neus;
        /// @brief 内部突触数量
        int syns_size;
        /// @brief 内部突触数据块
        SYNBlock syns;
        /// @brief 连接出的外部区域数量
        // int out_net_size;
        /// @brief 连接入的外部区域数量
        // int in_net_size;
        int npart;
        size_t* out_syn_size_list;
        int*    out_net_id_list;
        ADJBlock adjs;
    };
    
    /// @brief 仿真STDP模型时,所有设备地址信息
    struct CNetAddrs{

        std::vector<int*> clast_fired_addrs;
        std::vector<int*> csyn_src_addrs;
        std::vector<real*> csyn_weight_addrs;
    };
    struct GNetAddrs{
        int **glast_fired_addrs;
        int **gsyn_src_addrs;
        real **gsyn_weight_addrs;
    };
    
    struct MultiNet
    {
        std::vector<GSubNet *> cnets;
        std::vector<GSubNet *> gnets;
        std::vector<GNetAddrs> gaddrs;
        CNetAddrs caddrs;
        BufferManager manager;
        DenseBufferManager managerd;
        int max_delay;
        int min_delay;
        int blocksize;
        // int npart;
    };
    ///神经网络仿真数据操作

    void init_gsubnet_neus(GSubNet *cnet, int max_delay);
    void init_gsubnet_syns(GSubNet *cnet);
    void init_gsubnet_adjs(GSubNet *cnet,size_t net_axon_size,size_t net_dend_size);
    void free_gsubnet(GSubNet *cnet);
    void free_gsubnet_gpu(GSubNet *gnet);
    GSubNet *copy_subnet_gpu(GSubNet *cnet, int max_delay,CNetAddrs& addrs);
    
    size_t get_subnet_firecnt(GSubNet* cnet,GSubNet *gnet);
    void copy_subnet_cpu(GSubNet* gnet,GSubNet* cnet);
    ///脉冲缓冲区相关数据操作


    //Normal buffer

    SpikeBuffer* init_buffer_gpu(int size,SpikeBuffer& cbuffer);
    int* init_buffer_size_list_gpu(int size);
    void free_buffer_gpu(SpikeBuffer *gbuffer);
    SpikeBuffer **copy_buffers_gpu(std::vector<SpikeBuffer *> &cbuffers);


    //Dense buffer

    SpikeDenseBuffer* init_dense_buffer_gpu(int size,int max_delay,SpikeDenseBuffer* cbuffer,std::vector<int>& targets);
    void free_dense_buffer_gpu(SpikeDenseBuffer* gbuffer);
    SpikeDenseBuffer** copy_dense_buffers_gpu(std::vector<SpikeDenseBuffer*>&cbuffers);


    

    //NetAddrs相关GPU操作
    void copy_netaddrs_gpu(GNetAddrs& gaddrs,CNetAddrs& caddrs);
    void free_netaddrs_gpu(GNetAddrs& gaddrs);

    ///仿真常量相关的数据操作

    void copy_consts_gpu(int max_delay,real dt);
    void copy_consts_gpu(int max_delay, real dt,bool nlifconst, std::array<real, 30> lifconst,bool nstdpconst, std::array<real, 6> stdpconst);
    ///仿真核函数///

    /// @brief cuda核函数：仿真神经元
    /// @param net 
    /// @param step 
    /// @return 
    __global__ void mgsim_neus_core(GSubNet *net, int step);
    /// @brief cuda核函数：仿真突触（脉冲发放）
    /// @param net 
    /// @param step 
    /// @param buffer_size_list 
    /// @param buffers 
    /// @return 
    __global__ void mgsim_syns_core(GSubNet *net, int step, int *buffer_size_list, SpikeBuffer **buffers);
    /// @brief cuda核函数：仿真突触（stdp模型）
    /// @param net 
    /// @param step 
    /// @param last_fired_addrs 
    /// @param syn_src_addrs 
    /// @param syn_weight_addrs 
    /// @return 
    __global__ void mgsim_stdp_core(GSubNet *net, int step, int **last_fired_addrs, int **syn_src_addrs, real **syn_weight_addrs);
    /// @brief cuda核函数：接收脉冲
    /// @param net 
    /// @param index 
    /// @param buffers 
    /// @return 
    __global__ void mgsim_recv_core(GSubNet *net, int index, size_t buffer_size,SpikeBuffer **buffers);
    /// @brief cuda核函数:初始化泊松神经元随机种子
    /// @param size 
    /// @param state 
    /// @param seed 
    /// @return 
    __global__ void mgsim_init_core(int size,curandState *state, unsigned long seed);
    /// @brief cuda核函数：仿真突触 (fast)
    /// @param net 
    /// @param step 
    /// @param buffer_size_list 
    /// @param buffers 
    /// @return 
    __global__ void mgsim_syn_fast_core(GSubNet *net, int step, int *buffer_size_list, SpikeBuffer **buffers);
    /// @brief 接收域外密集脉冲
    /// @param net 
    /// @param index 
    /// @param neus_size 
    /// @param buffers 
    /// @return 
    __global__ void mgsim_recv_dense_core(GSubNet *net, int step,int index, size_t neus_size,SpikeDenseBuffer **buffers);
    /// @brief 仿真突触 （密集发放外部脉冲）
    /// @param net 
    /// @param step 
    /// @param buffer_size_list 
    /// @param buffers 
    /// @return 
    __global__ void mgsim_syn_fast_dense_core(GSubNet *net, int step, SpikeDenseBuffer **buffers);

    //传输和接收脉冲 仿真时间片（normal）

    void trans_spikes_sparse(GSubNet *cnet, BufferManager &manager, std::vector<int> &mapper, int turn, cudaStream_t &trans_stream);
    void recvs_spikes_sparse(GSubNet *gnet, GSubNet *cnet, int blocksize, BufferManager &manager, int turn, std::vector<cudaStream_t> &recv_streams);
    void mgsim_step_sparse(GSubNet *gnet, GSubNet *cnet, int step, int blocksize, BufferManager &manager,GNetAddrs& addrs, int turn,cudaStream_t& stream);

    //传输和接收脉冲 仿真时间片（dense）

    void trans_spikes_dense(GSubNet *cnet, DenseBufferManager &manager, int turn, cudaStream_t &trans_stream, std::vector<int> &mapper,int max_delay);
    void recvs_spikes_dense(GSubNet *gnet, GSubNet *cnet,int step, int blocksize, DenseBufferManager &manager, int turn, std::vector<cudaStream_t> &recv_streams);
    void mgsim_step_dense(GSubNet *gnet, GSubNet *cnet, int step, int blocksize, DenseBufferManager &manager,GNetAddrs& addrs, int turn);

    
    void test(int i);
    void test2(int * buffer,int size);

};