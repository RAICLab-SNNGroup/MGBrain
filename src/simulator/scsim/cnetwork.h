//
// Created by 15838 on 2023/3/6.
//
#pragma once
#ifndef SIMPLECPUSIM_NETWORK_HPP
#define SIMPLECPUSIM_NETWORK_HPP
#include "../../macro.h"
#include "cneuron.h"
#include "csynapse.h"
#include "cpop.h"
namespace MGBrain
{
    CBaseNeuron *createCNeuron(NeuronType type, int id, bool isSource)
    {
        CBaseNeuron *n;
        switch (type)
        {
        case NeuronType::LIF0:
            n = new CLIFNeuron(id, isSource);
            /* code */
            break;
        case NeuronType::POISSON:
            n = new CPoissonInNeuron(id);
            break;
        case NeuronType::LIF:
            n = new CLIFNeuron(id, isSource);
            break;
        default:
            n = nullptr;
            break;
        }
        return n;
    }

    class CNetwork
    {
    public:
        std::vector<Pop*> pops;
        std::vector<CBaseNeuron *> neurons;
        std::vector<CBaseSynapse *> synapses;
        int indexer = 0;
        CNetwork(){}
        ~CNetwork();
        void pushPop(Pop *p);
        int pushNeuron(NeuronType type, bool isSource);
        void pushSynapse(int src, int tar, real weight, real delay);
        CBaseNeuron &operator[](int index);
        CBaseNeuron *get(int pop, int index);
    };
};

#endif // SIMPLECPUSIM_NETWORK_HPP
