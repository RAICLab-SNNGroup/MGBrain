//
// Created by 15838 on 2023/3/6.
//

#ifndef SIMPLECPUSIM_SIMULATOR_HPP
#define SIMPLECPUSIM_SIMULATOR_HPP

#include "../../macro.h"
#include "cnetwork.h"

namespace MGBrain
{
    class SCSimulator
    {
    public:
        CNetwork *net;
        std::vector<real> output;
        real dt;
        void (*monitorn)(int, CBaseNeuron &);
        void (*monitors)(int, CBaseSynapse &);
        void (*monitorp)(int, int, std::vector<CBaseNeuron *>);

        SCSimulator(CNetwork *_net);
        ~SCSimulator();
        void setPoissonData(std::vector<real> data);
        void setMonitorNeuron(void (*_monitor)(int, CBaseNeuron &));
        void setMonitorSynapse(void (*_monitors)(int, CBaseSynapse &));
        void setMonitorPop(void (*_monitorp)(int, int, std::vector<CBaseNeuron *>));
        void simulate(real time);

    private:
        void simulate(int clock);
    };
};

#endif // SIMPLECPUSIM_SIMULATOR_HPP
