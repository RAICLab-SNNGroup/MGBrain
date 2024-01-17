//
// Created by 15838 on 2023/3/6.
//
#pragma once
#ifndef SIMPLECPUSIM_NETWORK_HPP
#define SIMPLECPUSIM_NETWORK_HPP
#include "../macro.h"
#include "model.h"
namespace MGBrain
{
    /// @brief 族群模型
    struct Pop
    {
        /// @brief 族群id
        int id;
        /// @brief 族群内神经元数量
        int num;
        /// @brief 是否是输入神经元
        bool source;
        /// @brief 神经元类型
        NeuronType type;
        /// @brief 神经元列表
        std::vector<int> neurons;
        Pop(int _id,int _num,bool _source,NeuronType _type):id(_id),num(_num),source(_source),type(_type){
            neurons.resize(_num);
        }
    };
    struct Neuron{
        int id;
        bool source;
        NeuronType type;
        /// @brief 神经元的激活概率
        real rate;
        int fire_cnt;
        std::vector<int> nxt;
        std::vector<int> pre;
        std::vector<size_t> syns;
        Neuron(int _id,bool _source,NeuronType _type):id(_id),source(_source),type(_type){
            rate=0;
            // nxts=0;
        }
    };
    struct Synapse{
        int src;
        int tar;
        real weight;
        real delay;
        Synapse(int _src,int _tar,real _weight,real _delay):src(_src),tar(_tar),weight(_weight),delay(_delay){}
    };
    class Network
    {
    public:
        // int nindexer = 0;
        // int sindexer = 0;
        std::vector<Pop> pops;
        std::vector<Neuron> neurons;
        std::vector<Synapse> synapses;

        std::array<real,30> lifconst;
        std::array<real,6> stdpconst;
        bool nlifconst=false;
        bool nstdpconst=false;
        // Network();
        // ~Network();
        // void push_pop(Population *p);
        // int push_neuron(NeuronType type, bool isSource);
        // void push_synapse(int src, int tar, real weight, real delay);
        // Neuron &operator[](int index);
        // Neuron *get(int pop, int index);
    };
}

#endif // SIMPLECPUSIM_NETWORK_HPP
