#pragma once
#include "mgstruct.h"
namespace MGBrain
{
    /// @brief 脉冲缓冲池
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
        std::vector<SpikeBuffer *> gout_buffers;
        std::vector<SpikeBuffer *> gin_buffers;
        std::vector<SpikeBuffer> cout_buffers;
        std::vector<SpikeBuffer> cin_buffers;
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
        std::vector<cudaStream_t> trans_streams;
        std::vector<cudaStream_t> sim_streams;
        std::vector<std::vector<cudaStream_t>> recv_streams;
        std::vector<std::vector<bool>> out_valid;
        std::vector<std::vector<bool>> in_valid;
        int net_num;
        void initBufferManager(int net_num)
        {
            // 初始化缓冲池
            this->net_num=net_num;
            netbuffers0.resize(net_num);
            netbuffers1.resize(net_num);
            for (int i = 0; i < net_num; i++)
            {
                netbuffers0[i].cbuffer_size_list = new int[net_num]();
                netbuffers1[i].cbuffer_size_list = new int[net_num]();

                netbuffers0[i].gin_buffers.resize(net_num, nullptr);
                netbuffers0[i].gout_buffers.resize(net_num, nullptr);
                netbuffers1[i].gin_buffers.resize(net_num, nullptr);
                netbuffers1[i].gout_buffers.resize(net_num, nullptr);

                netbuffers0[i].cout_buffers.resize(net_num);
                netbuffers1[i].cout_buffers.resize(net_num);
                netbuffers0[i].cin_buffers.resize(net_num);
                netbuffers1[i].cin_buffers.resize(net_num);
            }
            // 初始化缓冲池映射表
            mapperins.resize(net_num, std::vector<std::pair<int, int>>(net_num, {0, 0}));
            mapperouts.resize(net_num, std::vector<std::pair<int, int>>(net_num, {0, 0}));
            // 初始化缓冲池传输流
            trans_streams.resize(net_num);
            sim_streams.resize(net_num);
            recv_streams.resize(net_num, std::vector<cudaStream_t>());
            for (int i = 0; i < net_num; i++)
            {
                
                recv_streams[i].resize(net_num);
            }
            //
            out_valid.resize(net_num, std::vector<bool>(net_num, false));
            in_valid.resize(net_num, std::vector<bool>(net_num, false));
        }
        void clearBuffer(int netid, int turn,cudaStream_t& stream)
        {
            if (turn == 0)
            {
                CUDACHECK(cudaMemsetAsync(netbuffers0[netid].gbuffer_size_list, 0, sizeof(int) * net_num,stream));
            }
            else
            {
                CUDACHECK(cudaMemsetAsync(netbuffers1[netid].gbuffer_size_list, 0, sizeof(int) * net_num,stream));
            }
        }
        void syncBufferSizeList(int netid, int turn,cudaStream_t& stream)
        {
            if (turn == 0)
            {
                CUDACHECK(cudaMemcpyAsync(netbuffers0[netid].cbuffer_size_list, netbuffers0[netid].gbuffer_size_list, sizeof(int)*net_num, cudaMemcpyDeviceToHost,stream));
            }
            else
            {
                CUDACHECK(cudaMemcpyAsync(netbuffers1[netid].cbuffer_size_list, netbuffers1[netid].gbuffer_size_list, sizeof(int)*net_num, cudaMemcpyDeviceToHost,stream));
            }
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
                return netbuffers0[netid].cbuffer_size_list[offset];
            }
            else
            {
                std::tie(netid, offset) = mapperins[netid][offset];
                return netbuffers1[netid].cbuffer_size_list[offset];
            }
        }
        int getZones(){
            return netbuffers0.size();
        }
        int getTarNetId(int netid, int offset)
        {
            return mapperouts[netid][offset].first;
        }
        int getTarNetIdx(int netid, int offset)
        {
            return mapperouts[netid][offset].second;
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
        SpikeBuffer **getOutGGBuffer(int netid, int turn)
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
                return netbuffers0[netid].gin_buffers[offset];
            }
            else
            {
                std::tie(netid, offset) = mapperouts[netid][offset];
                return netbuffers1[netid].gin_buffers[offset];
            }
        }
        SpikeBuffer *getTarInGBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
            {
                std::tie(netid, offset) = mapperins[netid][offset];
                return netbuffers0[netid].gin_buffers[offset];
            }
            else
            {
                std::tie(netid, offset) = mapperins[netid][offset];
                return netbuffers1[netid].gin_buffers[offset];
            }
        }
        SpikeBuffer *getCurOutGBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
                return netbuffers0[netid].gout_buffers[offset];
            else
                return netbuffers1[netid].gout_buffers[offset];
        }
        SpikeBuffer *getCurInGBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
                return netbuffers0[netid].gin_buffers[offset];
            else
                return netbuffers1[netid].gin_buffers[offset];
        }

        SpikeBuffer &getTarOutCBuffer(int netid, int offset, int turn)
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
        SpikeBuffer &getTarInCBuffer(int netid, int offset, int turn)
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
        SpikeBuffer &getCurOutCBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
                return netbuffers0[netid].cout_buffers[offset];
            else
                return netbuffers1[netid].cout_buffers[offset];
        }
        SpikeBuffer &getCurInCBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
                return netbuffers0[netid].cin_buffers[offset];
            else
                return netbuffers1[netid].cin_buffers[offset];
        }
    };
};