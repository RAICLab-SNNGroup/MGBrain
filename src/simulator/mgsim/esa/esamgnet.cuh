#pragma once
#include "../mgstruct.h"
namespace MGBrain
{
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
        /// @brief 外部突触数量
        int out_net_size;
        // /// @brief 外部突触数据块
        OUTSYNBlock *outs;
        /// @brief 其他设备外部突触连接的数量
        int in_net_size;
        // ADJBlock adjs;
    };
    struct ADJBlock
    {
        size_t *axon_size;
        size_t *axon_offs;
        size_t *axon_refs;
        size_t *dend_size;
        size_t *dend_offs;
        size_t *dend_refs;
    };
    struct FiredBuffer
    {
        bool *fired;
        int *last_fired;
    };
    struct SpikeBuffer
    {
        /// @brief 脉冲发放的目标
        int *targets;
        /// @brief 脉冲发放的时间
        int *times;
        /// @brief 脉冲值
        real *spikes;
    };
    struct NetBuffers
    {
        int *cbuffer_size_list;
        int *gbuffer_size_list;
        SpikeBuffer **ggout_buffers;
        SpikeBuffer **ggin_buffers;
        std::vector<SpikeBuffer *> cout_buffers;
        std::vector<SpikeBuffer *> cin_buffers;
    };
    /// @brief 缓冲区管理
    struct BufferManager
    {
        /// 为每个子网络设置两个缓冲区，解决脉冲同步和仿真并行时访问脉冲数据冲突
        /// @brief 缓冲区0
        std::vector<NetBuffers> netbuffers0;
        /// @brief 缓冲区1
        std::vector<NetBuffers> netbuffers1;
        /// @brief 发放缓冲区和接收缓冲区的映射
        std::vector<std::vector<std::pair<int, int>>> mapperouts;
        std::vector<std::vector<std::pair<int, int>>> mapperins;
        std::vector<std::vector<cudaStream_t>> trans_streams;
        std::vector<std::vector<cudaStream_t>> recv_streams;
        void initBufferManager(int net_num)
        {
            netbuffers0.resize(net_num);
            netbuffers1.resize(net_num);
            mapperins.resize(net_num, std::vector<std::pair<int, int>>(net_num, {0, 0}));
            mapperouts.resize(net_num, std::vector<std::pair<int, int>>(net_num, {0, 0}));
        }
        void clearGBuffer(int netid, int turn)
        {
            if (turn == 0)
                CUDACHECK(cudaMemset(netbuffers0[netid].gbuffer_size_list, 0, sizeof(int) * netbuffers0[netid].cout_buffers.size()));
            else
                CUDACHECK(cudaMemset(netbuffers1[netid].gbuffer_size_list, 0, sizeof(int) * netbuffers1[netid].cout_buffers.size()));
        }
        void syncBufferSizeList(int netid, int turn)
        {
            if (turn == 0)
                CUDACHECK(cudaMemcpy(netbuffers0[netid].cbuffer_size_list, netbuffers0[netid].gbuffer_size_list, netbuffers0[netid].cout_buffers.size(), cudaMemcpyDeviceToHost));
            else
                CUDACHECK(cudaMemcpy(netbuffers1[netid].cbuffer_size_list, netbuffers1[netid].gbuffer_size_list, netbuffers1[netid].cout_buffers.size(), cudaMemcpyDeviceToHost));
        }
        int getOutBufferSize(int netid, int offset, int turn)
        {
            if (turn == 0)
                return netbuffers0[netid].cbuffer_size_list[offset];
            else
                return netbuffers1[netid].cbuffer_size_list[offset];
        }
        int getInBufferSize(int netid, int offset, int turn)
        {
            if (turn == 0)
            {
                std::tie(netid, offset) = mapperins[netid][offset];
                return netbuffers1[netid].cbuffer_size_list[offset];
            }
            else
            {
                std::tie(netid, offset) = mapperins[netid][offset];
                return netbuffers1[netid].cbuffer_size_list[offset];
            }
        }
        int getTarNetId(int netid, int offset)
        {
            return mapperouts[netid][offset].first;
        }
        SpikeBuffer **getOutBuffer(int netid, int turn)
        {
            if (turn == 0)
            {
                return netbuffers0[netid].ggout_buffers;
            }
            else
            {
                return netbuffers1[netid].ggout_buffers;
            }
        }
        SpikeBuffer **getInGGBuffer(int netid, int turn)
        {
            if (turn == 0)
            {
                return netbuffers0[netid].ggin_buffers;
            }
            else
            {
                return netbuffers1[netid].ggin_buffers;
            }
        }
        SpikeBuffer *getTarOutGBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
            {
                std::tie(netid, offset) = mapperouts[netid][offset];
                return netbuffers0[netid].cin_buffers[offset];
            }
            else
            {
                std::tie(netid, offset) = mapperouts[netid][offset];
                return netbuffers1[netid].cin_buffers[offset];
            }
        }
        SpikeBuffer *getTarInGBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
            {
                std::tie(netid, offset) = mapperouts[netid][offset];
                return netbuffers0[netid].cin_buffers[offset];
            }
            else
            {
                std::tie(netid, offset) = mapperouts[netid][offset];
                return netbuffers1[netid].cin_buffers[offset];
            }
        }
        SpikeBuffer *getCurOutGBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
                return netbuffers0[netid].cout_buffers[offset];
            else
                return netbuffers1[netid].cout_buffers[offset];
        }
        SpikeBuffer *getCurInGBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
                return netbuffers0[netid].cin_buffers[offset];
            else
                return netbuffers1[netid].cin_buffers[offset];
        }
        NetBuffers &getNetBuffers(int netid, int turn)
        {
            if (turn == 0)
                return netbuffers0[netid];
            else
                return netbuffers1[netid];
        }
        int *getGBufferSizeList(int netid, int turn)
        {
            if (turn == 0)
                return netbuffers0[netid].gbuffer_size_list;
            else
                return netbuffers1[netid].gbuffer_size_list;
        }
    };
    struct MultiNet
    {
        std::vector<GSubNet *> cnets;
        std::vector<GSubNet *> gnets;
        BufferManager manager;
        int max_delay;
        int min_delay;
        int blocksize;
    };
    void init_gsubnet_neus(GSubNet *cnet, int max_delay);
    void init_gsubnet_syns(GSubNet *cnet);
    void initGSubNetOutSyns(SYNBlock *block, int num);
    void free_gsubnet(GSubNet *cnet);
    void free_gsubnet_gpu(GSubNet *gnet);
    GSubNet *copy_subnet_gpu(GSubNet *cnet, int max_delay);
    void init_buffer(SpikeBuffer *cbuffer, int size);
    void free_buffer(SpikeBuffer *cbuffer);
    void free_buffer_gpu(SpikeBuffer *gbuffer);
    SpikeBuffer *copy_buffer_gpu(SpikeBuffer *cbuffer, int size);
    SpikeBuffer **copy_buffers_gpu(std::vector<SpikeBuffer *> &cbuffers);
    void copy_consts_gpu(int max_delay, LIFConsts *lifconst = nullptr, STDPConsts *stdpconst = nullptr);

    __global__ void mgsimReloc(GSubNet *net, int step);
    __global__ void mgsim_neus_core(GSubNet *net, int step);
    __global__ void mgsimSyns(GSubNet *net, int step);
    // __global__ void mgsimOuts(GSubNet *net, int step, int index, real *spikes, int *targets, int *times, int *out_buffer_sizes);
    __global__ void mgsim_recv_core(GSubNet *net, int index, SpikeBuffer **buffers);
    void transfer(SpikeBuffer *out_buffer, int out_device, SpikeBuffer *in_buffer, int in_device, int size, cudaStream_t stream);
    void mgsim_step(GSubNet *gnet, GSubNet *cnet, int step, int blocksize, BufferManager &manager, int turn);
    void trans_spikes(GSubNet *cnet, BufferManager &manager, int turn, std::vector<cudaStream_t> &trans_streams, std::vector<int> &mapper);
    void recvs_spikes(GSubNet *gnet, GSubNet *cnet, int blocksize, BufferManager &manager, int turn, std::vector<cudaStream_t> &recv_streams);
};