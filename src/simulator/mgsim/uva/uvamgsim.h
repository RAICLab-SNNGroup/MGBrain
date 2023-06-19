#pragma once
#include "uvamgnet.cuh"
namespace MGBrain{
    class MGSimulatorUVA
    {
    private:
        /// @brief UVA方式构建的多区域脉冲神经网络
        MultiNetUVA multinet;
        /// @brief 子网络号与设备号的映射
        std::vector<int> mapper;
        // std::unordered_map<int,int> mapper;
        
        /// @brief 构建网络
        /// @param net 神经网络
        /// @param part 划分结果
        /// @param npart 划分区域数量
        /// @return 构建的子网络
        void build_multinet_uva(Network &net, std::vector<int> part, int npart);
        /// @brief 将网络从主机内存拷贝到对应GPU中
        void copy_gnetuva_to_gpu();
        /// @brief 查询可用的GPU设备
        /// @param n 需要的设备数量
        void query_device();
    public:
        /// @brief 生成多GPU仿真器
        /// @param net 网络结构
        /// @param part 划分结果
        /// @param nparts 划分区域数量
        MGSimulatorUVA(Network &net, std::vector<int> part,int nparts);
        ~MGSimulatorUVA();
        /// @brief 开始仿真
        /// @param time 仿真时间
        void simulate(real time);
    };
};