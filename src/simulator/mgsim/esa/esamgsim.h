#pragma once
#include "esamgnet.cuh"

namespace MGBrain{
    class MGSimulator{
        /// @brief 构建的多区域脉冲神经网络
        MultiNet multinet;
        /// @brief 子网络号与设备号的映射
        std::vector<int> mapper;
        /// @brief 构建网络
        /// @param net 神经网络
        /// @param part 划分结果
        /// @param npart 划分区域数量
        /// @return 构建的子网络
        void build_multinet(Network &net, std::vector<int> part, int npart);
        /// @brief 将网络从主机内存拷贝到对应GPU中
        void copy_gnets_to_gpu();
        /// @brief 查询可用的GPU设备
        /// @param n 需要的设备数量
        void query_device();
        void simulate_half_group(int netid,int step,int half_group_size,int turn);
        size_t gen_syn_ref(size_t zone,size_t offset){
            return offset|(zone<<56);
        }
    public:
        /// @brief 生成多GPU仿真器
        /// @param net 网络结构
        /// @param part 划分结果
        /// @param nparts 划分区域数量
        
        MGSimulator(Network &net, std::vector<int> part,int nparts);
        ~MGSimulator();
        /// @brief 开始仿真
        /// @param time 仿真时间
        void simulate(real time);

    };
}