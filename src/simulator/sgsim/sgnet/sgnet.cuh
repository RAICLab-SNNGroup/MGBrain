#pragma once
#include "../sgstruct.h"
namespace MGBrain
{
    struct GNetwork
    {
        int neus_size;
        NEUBlock neus;
        int syns_size;
        SYNBlock syns;
    };
    void initNeus(GNetwork* cnet,int max_delay);
    void initSyns(GNetwork* cnet);
    void freeNet(GNetwork* cnet);
    /// @brief 初始化LIF神经元常量
    /// @param consts 
    void initLIFConsts(LIFConsts* consts);
    /// @brief 初始化突触常量
    /// @param consts 
    void initSTDPConsts(STDPConsts* consts);
    void init_gpu_consts(int max_delay,LIFConsts* lifconst=nullptr,STDPConsts* stdpconst=nullptr);

    /// @brief 把网络数据拷贝到GPU中
    /// @param cnet 
    /// @return 
    GNetwork* copyNetToGPU(GNetwork* cnet,int max_delay);
    void gpuFreeNet(GNetwork* gnet);
    /// @brief CUDA核函数 仿真神经元
    /// @param net snn网络
    /// @param step 当前时间片
    /// @return 
    __global__ void simNeus(GNetwork *net, int step);
    /// @brief CUDA核函数 仿真突触
    /// @param net 
    /// @param step 
    /// @return 
    __global__ void simSyns(GNetwork *net, int step);
    /// @brief 仿真一个时间片
    /// @param gnet GNetwork网络
    /// @param step 
    /// @param blocksize 
    /// @param neunum 
    /// @param synnum 
    void simStep(GNetwork *gnet,int step,int blocksize,int neunum,int synnum);
};
