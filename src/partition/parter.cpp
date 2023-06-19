#include "parter.h"

void MGBrain::Partitioner::part_with_net(Network &network, int nparts, std::vector<idx_t> &part)
{
    int vtx_num = network.neurons.size();
    size_t edge_num = network.synapses.size();
    std::vector<idx_t> adjx(vtx_num + 1);
    std::vector<idx_t> adjn(2 * edge_num);
    adjx[0] = 0;
    int k = 0;
    for (int i = 0; i < network.neurons.size(); i++)
    {
        for (int j = 0; j < network.neurons[i].pre.size(); j++)
        {
            adjn[k] = network.neurons[i].pre[j];
            k++;
        }
        for (int j = 0; j < network.neurons[i].nxt.size(); j++)
        {
            adjn[k] = network.neurons[i].nxt[j];
            k++;
        }
        adjx[i + 1] = k;
    }
    MetisUtil::metis_part_normal_graph(vtx_num, nparts, adjx, adjn, part);
}
void MGBrain::Partitioner::part_with_model(Network &network, int nparts, std::vector<int> &part)
{
    int vtx_num = network.neurons.size();
    size_t edge_num = network.synapses.size();
    std::vector<idx_t> adjx(vtx_num + 1, 0);
    std::vector<idx_t> adjn(2 * edge_num);
    std::vector<idx_t> ewgt(2 * edge_num);
    std::vector<idx_t> vwgt(vtx_num);
    float neu_rate = 0.5;
    float syn_rate = 0.01;
    for (int i = 0, t = 0; i < network.neurons.size(); i++)
    {
        size_t presize = network.neurons[i].pre.size();
        size_t nxtsize = network.neurons[i].nxt.size();
        float edgewgts = 0;
        for (size_t j = 0; j < presize; j++)
        {
            adjn[t] = network.neurons[i].pre[j];
            // ewgt[t] = network.neurons[i]->adj.size();
            ewgt[t]=1;
            t++;
        }
        for(size_t j=0;j<nxtsize;j++){
            adjn[t] = network.neurons[i].nxt[j];
            // ewgt[t] = network.neurons[i]->adj.size();
            ewgt[t]=1;
            t++;
        }
        for (size_t j = 0; j < nxtsize; j++)
        {
            edgewgts += ModelCalculator::synapseWeight(SynapseType::STATIC, syn_rate);
        }
        vwgt[i] = ModelCalculator::neuronWeight(network.neurons[i].type, neu_rate, edgewgts);
        adjx[i + 1] = t;
    }
    MetisUtil::metis_part_weighted_graph(vtx_num, nparts, adjx, adjn, vwgt, ewgt, part);
}
void MGBrain::Partitioner::part_with_bsim(Network &network, int nparts, std::vector<int> &part)
{
    int vtx_num = network.neurons.size();
    size_t edge_num = network.synapses.size();
    size_t capacity = std::ceil( edge_num/ nparts);
    size_t cursize = 0;
    int p = 0;
    for (int i = 0; i < vtx_num; i++)
    {
        cursize += network.neurons[i].nxt.size();
        if (cursize <= capacity)
        {
            part[i] = p;
        }
        else
        {
            cursize = 0;
            if(p<nparts-1)
            p++;
        }
    }
}
void MGBrain::Partitioner::part_with_simple(Network &network, int nparts, std::vector<int> &part)
{
    int vtx_num = network.neurons.size();
    int edge_num = network.synapses.size();
    int zone_size = vtx_num / nparts;
    int rest = vtx_num % nparts;
    for (int p = 0, i = rest; p < nparts; p++)
    {
        for (int k = 0; k < zone_size; k++, i++)
        {
            part[i] = p;
        }
    }
}
std::vector<int> MGBrain::Partitioner::part_network(Network &network, int nparts, PartType type)
{
    std::vector<int> part(network.neurons.size(), 0);
    if (nparts <= 1)
        return part;
    switch (type)
    {
    case PartType::METIS:
        part_with_net(network, nparts, part);
        break;
    case PartType::MODEL:
        part_with_model(network, nparts, part);
        break;
    case PartType::BSIM:
        part_with_bsim(network, nparts, part);
        break;
    case PartType::SIMPLE:
        part_with_simple(network, nparts, part);
        break;
    default:
        break;
    }
    return part;
}