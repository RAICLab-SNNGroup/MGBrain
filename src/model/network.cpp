
#include "network.h"
// MGBrain::Network::Network(){
    
// }
// MGBrain::Network::~Network()
// {
// }
// void MGBrain::Network::push_pop(Population *p)
// {
//     pops.push_back(p);
// };
// int MGBrain::Network::push_neuron(NeuronType type, bool isSource)
// {
//     int id = nindexer++;
//     Neuron *n=new Neuron();
//     n->id=id;
//     n->source=isSource;
//     n->type=type;
//     n->rate=0;
//     neurons.push_back(n);
//     return n->id;
// }
// void MGBrain::Network::push_synapse(int src, int tar, real weight, real delay)
// {
//     neurons[src]->adj.push_back(tar);
//     neurons[tar]->adj.push_back(src);
//     // neurons[src]->nxt.push_back(tar);
//     neurons[src]->nxts++;
//     auto *syn = new Synapse();
//     syn->tar = tar;
//     syn->src = src;
//     syn->weight = weight;
//     syn->delay = delay;
//     synapses.push_back(syn);
// }
// MGBrain::Neuron& MGBrain::Network::operator[](int index)
// {
//     return *neurons[index];
// }
// MGBrain::Neuron* MGBrain::Network::get(int pop, int index)
// {
//     for (auto p : pops)
//     {
//         if (p->id == pop)
//         {
//             int offset = p->neurons[index];
//             return neurons[offset];
//         }
//     }
//     return nullptr;
// }