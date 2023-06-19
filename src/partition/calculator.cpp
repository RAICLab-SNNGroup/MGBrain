#include "calculator.h"
namespace MGBrain{
    std::unordered_map<NeuronType,int> ModelCalculator::neuwgt={
            {NeuronType::LIF,8},// lif模型计算开销
            {NeuronType::POISSON,2},// poisson模型计算开销
            {NeuronType::LIF0,4},
            {NeuronType::RAW,1},
            {NeuronType::LIFB,4},
            {NeuronType::LIFE,6}
        };
    std::unordered_map<SynapseType,int> ModelCalculator::synwgt={
            {SynapseType::STATIC,2}, // 静态突触模型计算开销
            {SynapseType::STDP,6} // stdp突触模型计算开销
        };
};