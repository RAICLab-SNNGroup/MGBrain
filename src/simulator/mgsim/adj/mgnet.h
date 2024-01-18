#pragma once
#include "mgstruct.h"
#include "mgcore.cuh"
namespace MGBrain
{
    /// @brief 仿真STDP模型时,所需的跨设备地址信息
    struct CNetAddrs{
        std::vector<int*> clast_fired_addrs;
        std::vector<int*> csyn_src_addrs;
        std::vector<real*> csyn_weight_addrs;
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
    void init_gsubnet_neus(GSubNet *cnet,int num, int max_delay);
    void init_gsubnet_syns(GSubNet *cnet,int num);
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

    
}