//
// Created by 15838 on 2023/3/6.
//
#pragma once
#ifndef SIMPLECPUSIM_NETGEN_HPP
#define SIMPLECPUSIM_NETGEN_HPP
#include "../macro.h"
#include "model.h"
#include "network.h"
namespace MGBrain
{
    class NetGen
    {
    public:
        static void rand_vector(std::vector<bool> &list, int gen_num, int max_num);
        static real rand_weight(real floor, real ceil);
        static real rand_delay(real floor, real ceil);
        static void gen_net(Network& net,Model& model);
        // static Network &gen_net_mp(Model *model);
    };
}

#endif // SIMPLECPUSIM_NETGEN_HPP
