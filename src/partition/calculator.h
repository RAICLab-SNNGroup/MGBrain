#pragma once
#include "../macro.h"
namespace MGBrain
{
    class ModelCalculator
    {
    public: 
        static std::unordered_map<NeuronType,int> neuwgt;
        static std::unordered_map<SynapseType,int> synwgt;   
        static inline int neuronWeight(NeuronType neutype,float neu_rate,float synwgts)
        {
            float vtx_wgt = 1+ neuwgt[neutype] * neu_rate+synwgts;
            return std::round(vtx_wgt);
        }
        static inline float synapseWeight(SynapseType syntype,float syn_rate) {
            return synwgt[syntype]*syn_rate;
        }
    };
};