#include "csimulator.h"



MGBrain::SCSimulator::SCSimulator(CNetwork *_net)
{
    net = _net;
    dt = Config::STEP;
    // 初始化突触脉冲缓冲池
    for (auto &synapse : net->synapses)
    {
        int size = std::lrint(synapse->delay / dt);
        synapse->spikes.initialize(size, 0);
    }

    monitorn = [](int clock, CBaseNeuron &n) {};
    monitors = [](int clock, CBaseSynapse &s) {};
    monitorp = [](int clock, int pop, std::vector<CBaseNeuron *> ns) {};
}
MGBrain::SCSimulator::~SCSimulator()
{
    output.clear();
}
void MGBrain::SCSimulator::setPoissonData(std::vector<real> data)
{

    Pop *tp = nullptr;
    for (auto &p : net->pops)
    {
        if (p->source)
        {
            tp = p;
        }
    }
    if (tp == nullptr)
        return;
    for (int i = 0; i < tp->neurons.size(); i++)
    {
        (*net)[tp->neurons[i]].recv(data[i]);
    }
}
void MGBrain::SCSimulator::setMonitorNeuron(void (*_monitor)(int, CBaseNeuron &))
{
    monitorn = _monitor;
}
void MGBrain::SCSimulator::setMonitorSynapse(void (*_monitors)(int, CBaseSynapse &))
{
    monitors = _monitors;
}
void MGBrain::SCSimulator::setMonitorPop(void (*_monitorp)(int, int, std::vector<CBaseNeuron *>))
{
    monitorp = _monitorp;
}
void MGBrain::SCSimulator::simulate(real time)
{
    int steps = std::lrint(time / dt);
    for (int i = 0; i < steps; i++)
    {
        // feeder(i, *inputs);
        simulate(i);
        // inputs->clear();
    }
}
void MGBrain::SCSimulator::simulate(int clock)
{
    // // 准备输入脉冲
    // for (int i = 0; i < inputs->size(); i++)
    // {
    //     net->source[i]->recv((*inputs)[i]);
    // }
    // 仿真神经元
    for (auto p : net->pops)
    {
        std::vector<CBaseNeuron *> ns;
        for (auto i : p->neurons)
        {
            // 更新神经元
            (*net)[i].update(clock);
            ns.push_back((*net).neurons[i]);
            monitorn(clock, (*net)[i]);
        }
        monitorp(clock, p->id, ns);
        for (auto i : p->neurons)
        {
            // 清楚神经元的输入电流
            (*net)[i].clear();
        }
    }
    // 仿真突触
    for (auto synapse : net->synapses)
    {
        real o = synapse->update(*(net->neurons[synapse->src]), *(net->neurons[synapse->tar]));
        monitors(clock, *synapse);
        net->neurons[synapse->tar]->recv(o);
        // net->neurons[synapse->tar]->addIn(o);
    }
    // std::cout<<net->neurons[4]->in<<std::endl;
}
