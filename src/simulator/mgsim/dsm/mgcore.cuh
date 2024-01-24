#pragma once
#include "mgstruct.h"
namespace MGBrain
{

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

    // 初始化随机种子
    void init_state(int num,curandState* gstates);

    
    void test(int i);
    void test2(int * buffer,int size);


    ///仿真常量相关的数据操作
    void copy_consts_gpu(int max_delay,real dt);
    void copy_consts_gpu(int max_delay, real dt,bool nlifconst, std::array<real, 30> lifconst,bool nstdpconst, std::array<real, 6> stdpconst);

};