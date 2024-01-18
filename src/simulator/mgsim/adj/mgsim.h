#pragma once
#include "mgnet.h"
namespace MGBrain{
    class MGSimulator{
        /// @brief 构建的多区域脉冲神经网络
        MultiNet multinet;
        /// @brief 网络号与设备号的映射
        std::vector<int> mapper;
        double sim_time;
        /// @brief 构建网络
        /// @param net 神经网络
        /// @param part 划分结果
        /// @param npart 划分区域数量
        /// @return 构建的子网络
        void build_multinet(Network &net, std::vector<int>& part, int npart);
        /// @brief 将网络从主机内存拷贝到对应GPU中
        void copy_gnets_to_gpu(Network& net);
        /// @brief 生成一般缓冲池
        void gen_normal_buffer();
        /// @brief 生成密集缓冲池
        void gen_dense_buffer(std::vector<std::vector<std::vector<int>>>& out);
        /// @brief  释放一般缓冲池
        void free_normal_buffer();
        /// @brief 释放密集缓冲池
        void free_dense_buffer();
        /// @brief 查询可用的GPU设备
        /// @param n 需要的设备数量
        void query_device();
        void simulate_half_group(int netid,int step,int half_group_size,int turn);
        size_t gen_syn_ref(size_t zone,size_t offset){
            return offset|(zone<<56);
        }
        int get_syn_zone(size_t ref){
            return (ref & (0xff00000000000000)) >> 56;
        }
        size_t get_syn_offset(size_t ref){
            return (ref & (0x00ffffffffffffff));
        }
        real get_firing_rate(real time);

        void simulate_sparse(real time);
        void simulate_dense(real time);
        void simulate_seq_sparse(real time);
        void simulate_seq_dense(real time);
        void simulate_test_d(real time);
        void simulate_test_s(real time);
    public:
        /// @brief 生成多GPU仿真器
        /// @param net 网络结构
        /// @param part 划分结果
        /// @param nparts 划分区域数量
        
        MGSimulator(Network &net, std::vector<int>& part,int nparts);
        ~MGSimulator();
        /// @brief 开始仿真
        /// @param time 仿真时间
        void simulate(real time);
        double get_time();

    };
}