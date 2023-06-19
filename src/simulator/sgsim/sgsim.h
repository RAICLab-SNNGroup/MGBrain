#pragma once

#include "../../model/network.h"

#include "./sgnet/sgnet.cuh"

namespace MGBrain
{
    class SGSimultor
    {
    private:
        GNetwork* cnet;
        GNetwork* gnet;
        int max_delay;
    public:
        SGSimultor(Network& net);
        ~SGSimultor();
        void simulate(real time);
        void build_gnetwork(Network &net);
    };
};
