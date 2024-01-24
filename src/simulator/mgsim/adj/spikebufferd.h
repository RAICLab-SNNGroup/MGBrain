#pragma once
#include "mgstruct.h"
namespace MGBrain
{
    /// @brief 脉冲缓冲池
    struct SpikeDenseBuffer
    {
        int start;
        int end;
        int neusize;
        /// @brief ID与offset映射
        int *mapper;
        /// @brief 脉冲发放的目标
        int *targets;
        /// @brief 脉冲值
        real *buffer_exc;
        real *buffer_inh;
    };
    struct NetDenseBuffers
    {
        // std::vector<int> cout_neus;
        // std::vector<int> cin_neus;
        SpikeDenseBuffer **ggout_buffers;
        SpikeDenseBuffer **ggin_buffers;
        std::vector<SpikeDenseBuffer *> gout_buffers;
        std::vector<SpikeDenseBuffer *> gin_buffers;
        std::vector<SpikeDenseBuffer*> cout_buffers;
        std::vector<SpikeDenseBuffer*> cin_buffers;
    };
    /// @brief 缓冲区管理
    struct DenseBufferManager
    {
        /// 为每个子网络设置两个缓冲区，解决脉冲同步和仿真并行时访问脉冲数据冲突
        /// @brief 缓冲区0
        std::vector<NetDenseBuffers> netbuffers0;
        /// @brief 缓冲区1
        std::vector<NetDenseBuffers> netbuffers1;
        /// @brief 发放缓冲区和接收缓冲区的映射
        std::vector<std::vector<int>> out_neu_sizes;
        std::vector<std::vector<std::pair<int, int>>> mapperouts;
        std::vector<std::vector<std::pair<int, int>>> mapperins;
        std::vector<cudaStream_t> trans_streams;
        std::vector<std::vector<cudaStream_t>> recv_streams;
        std::vector<cudaStream_t> sim_streams;
        std::vector<std::vector<bool>> out_valid;
        std::vector<std::vector<bool>> in_valid;
        int net_num;
        int max_delay;
        void initBufferManager(int net_num,int max_delay)
        {
            // 初始化缓冲池
            this->net_num=net_num;
            this->max_delay=max_delay;
            netbuffers0.resize(net_num);
            netbuffers1.resize(net_num);
            out_neu_sizes.resize(net_num);
            for (int i = 0; i < net_num; i++)
            {
                // netbuffers0[i].cout_neus = std::vector<int>(net_num);
                // netbuffers1[i].cout_neus = std::vector<int>(net_num);
                out_neu_sizes[i].resize(net_num,0);
                // netbuffers1[i].cin_neus = std::vector<int>(net_num);

                netbuffers0[i].gin_buffers.resize(net_num, nullptr);
                netbuffers0[i].gout_buffers.resize(net_num, nullptr);
                netbuffers1[i].gin_buffers.resize(net_num, nullptr);
                netbuffers1[i].gout_buffers.resize(net_num, nullptr);

                netbuffers0[i].cout_buffers.resize(net_num,nullptr);
                netbuffers1[i].cout_buffers.resize(net_num,nullptr);
                netbuffers0[i].cin_buffers.resize(net_num,nullptr);
                netbuffers1[i].cin_buffers.resize(net_num,nullptr);
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
        void clearBuffer(int netid, int turn)
        {
            if (turn == 0)
            {
                for(int i=0;i<net_num;i++){
                    int size=out_neu_sizes[netid][i];
                    if(size>0){
                        CUDACHECK(cudaMemset(netbuffers0[netid].gout_buffers[i], 0.0f, sizeof(real) * size*max_delay));
                    }
                }
            }
            else
            {
                for(int i=0;i<net_num;i++){
                    int size=out_neu_sizes[netid][i];
                    if(size>0){
                        CUDACHECK(cudaMemset(netbuffers1[netid].gout_buffers[i], 0.0f, sizeof(real) * size*max_delay));
                    }
                }
            }
        }
        int getOutNeuSize(int netid, int offset)
        {
            return out_neu_sizes[netid][offset];
        }
        int getInNeuSize(int netid, int offset)
        {
            std::tie(netid, offset) = mapperins[netid][offset];
            return out_neu_sizes[netid][offset];
        }
        int getZones(){
            return net_num;
        }
        int getTarNetId(int netid, int offset)
        {
            return mapperouts[netid][offset].first;
        }
        int getTarNetIdx(int netid, int offset)
        {
            return mapperouts[netid][offset].second;
        }
        SpikeDenseBuffer **getOutGGBuffer(int netid, int turn)
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
        SpikeDenseBuffer **getInGGBuffer(int netid, int turn)
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
        SpikeDenseBuffer* getTarInCBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
            {
                int tar_net= mapperouts[netid][offset].first;
                int tar_idx=mapperouts[netid][offset].second;
                return netbuffers0[tar_net].cin_buffers[tar_idx];
            }
            else
            {
                int tar_net= mapperouts[netid][offset].first;
                int tar_idx=mapperouts[netid][offset].second;
                return netbuffers1[tar_net].cin_buffers[tar_idx];
            }
        }
        SpikeDenseBuffer* getCurOutCBuffer(int netid, int offset, int turn)
        {
            if (turn == 0)
                return netbuffers0[netid].cout_buffers[offset];
            else
                return netbuffers1[netid].cout_buffers[offset];
        }
    };
};