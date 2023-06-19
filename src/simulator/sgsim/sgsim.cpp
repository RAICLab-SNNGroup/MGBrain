#include "sgsim.h"
/// SGSimulator
MGBrain::SGSimultor::SGSimultor(Network &net)
{
    build_gnetwork(net);
    init_gpu_consts(max_delay,nullptr,nullptr);
    gnet = copyNetToGPU(cnet,max_delay);
}
MGBrain::SGSimultor::~SGSimultor()
{
    freeNet(cnet);
    gpuFreeNet(gnet);
}
void MGBrain::SGSimultor::simulate(real time)
{
    int steps = time / Config::STEP;
    cudaSetDevice(0);
    // printf("start simulate\n");
    for (int i = 0; i < steps; i++)
    {
        simStep(gnet, i, 1024, cnet->neus_size, cnet->syns_size);
    }
    // printf("simulate complete\n");
}

void MGBrain::SGSimultor::build_gnetwork(Network &net)
{
    int neu_size = net.neurons.size();
    int syn_size = net.synapses.size();
    cnet = new GNetwork();
    cnet->neus_size = neu_size;
    cnet->syns_size = syn_size;
    initSyns(cnet);
    for (int i = 0; i < net.synapses.size(); i++)
    {
        cnet->syns.src[i] = net.synapses[i].src;
        cnet->syns.tar[i] = net.synapses[i].tar;
        cnet->syns.delay[i] = net.synapses[i].delay / Config::STEP;
        max_delay = std::max(max_delay, cnet->syns.delay[i]);
        cnet->syns.weight[i] = net.synapses[i].weight;
    }
    initNeus(cnet,max_delay);
    for (int i = 0; i < net.neurons.size(); i++)
    {
        if (net.neurons[i].type == NeuronType::POISSON)
        {
            cnet->neus.ids[i]=i;
            cnet->neus.poisson[i] = true;
            cnet->neus.rate[i] = net.neurons[i].rate;
        }
    }
}


