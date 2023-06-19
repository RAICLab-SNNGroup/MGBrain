
#include "cnetwork.h"
MGBrain::CNetwork::~CNetwork()
{
    for (auto &neuron : neurons)
    {
        delete neuron;
    }
    for (auto &synapse : synapses)
    {
        delete synapse;
    }
}
void MGBrain::CNetwork::pushPop(Pop *p)
{
    pops.push_back(p);
};
int MGBrain::CNetwork::pushNeuron(NeuronType type, bool isSource)
{
    int id = indexer++;
    CBaseNeuron *n=createCNeuron(type,id,isSource);
    return n->getId();
}
void MGBrain::CNetwork::pushSynapse(int src, int tar, real weight, real delay)
{
    neurons[src]->adj.push_back(tar);
    neurons[tar]->adj.push_back(src);
    auto *syn = new CBaseSynapse();
    syn->tar = tar;
    syn->src = src;
    syn->weight = weight;
    syn->delay = delay;
    synapses.push_back(syn);
}
MGBrain::CBaseNeuron& MGBrain::CNetwork::operator[](int index)
{
    return *neurons[index];
}
MGBrain::CBaseNeuron* MGBrain::CNetwork::get(int pop, int index)
{
    for (auto p : pops)
    {
        if (p->id == pop)
        {
            int offset = p->neurons[index];
            return neurons[offset];
        }
    }
    return nullptr;
}