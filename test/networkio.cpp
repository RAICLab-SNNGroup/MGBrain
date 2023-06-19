#include "../model/network.h"
namespace MGBrain
{
    struct pop_data
    {
        int id;
        int num;
        bool source;
        short type;
    };
    struct neuron_data
    {
        int id;
        short type;
        bool source;
        real rate;
    };
    struct synapse_data
    {
        int src;
        int tar;
        real weight;
        real delay;
    };
    void save_network(std::string filename, Network &net)
    {
        std::ofstream ofile(filename, std::ios::out | std::ios::binary);
        if (ofile.is_open())
        {
            // 写族群
            int psize = net.pops.size();
            ofile.write((char *)&psize, sizeof(int));
            for (int i = 0; i < psize; i++)
            {
                Population *p = net.pops[i];
                pop_data pd = {p->id, p->num, p->source, p->type};
                ofile.write((char *)&pd, sizeof(pop_data));
                int *narr = new int[p->num];
                for (int j = 0; j < p->num; j++)
                {
                    narr[j] = p->neurons[j];
                }
                ofile.write((char *)&narr, sizeof(int) * p->num);
            }
            // 写神经元
            int nsize = net.neurons.size();
            ofile.write((char *)&nsize, sizeof(int));
            for (int i = 0; i < nsize; i++)
            {
                Neuron *n = net.neurons[i];
                neuron_data nd = {n->id, n->type, n->source,n->rate};
                ofile.write((char *)&nd, sizeof(neuron_data));
            }
            // 写突触
            int ssize = net.synapses.size();
            ofile.write((char *)&ssize, sizeof(int));
            for (int i = 0; i < ssize; i++)
            {
                Synapse *s = net.synapses[i];
                synapse_data sd = {s->src, s->tar, s->weight, s->delay};
                ofile.write((char *)&sd, sizeof(synapse_data));
            }
        }
        ofile.close();
    };

    void load_network(std::string filename, Network &net){
        net.~Network();
        std::ifstream rfile(filename, std::ios::in | std::ios::binary);
        if (rfile.is_open())
        {
            // 读族群
            int psize;
            rfile.read((char*)&psize,sizeof(int));
            net.pops.resize(psize);
            for (int i = 0; i < psize; i++)
            {
                // Population *p = net.pops[i];
                pop_data pd;
                rfile.read((char *)&pd, sizeof(pop_data));
                net.pops[i]=new Population();
                net.pops[i]->id=pd.id;
                net.pops[i]->num=pd.num;
                net.pops[i]->source=pd.source;
                net.pops[i]->type=(NeuronType)pd.type;
                int *narr = new int[pd.num];
                rfile.read((char *)&narr, sizeof(int) * pd.num);
                net.pops[i]->neurons.resize(pd.num);
                for (int j = 0; j < pd.num; j++)
                {
                    net.pops[i]->neurons[j]=narr[j];
                }
                
            }
            // 读神经元
            int nsize;
            rfile.read((char *)&nsize, sizeof(int));
            net.neurons.resize(nsize);
            for (int i = 0; i < nsize; i++)
            {
                neuron_data nd;
                rfile.read((char *)&nd, sizeof(neuron_data));
                net.neurons[i]=new Neuron();
                net.neurons[i]->id=nd.id;
                net.neurons[i]->type=(NeuronType)nd.type;
                net.neurons[i]->source=nd.type;
                net.neurons[i]->rate=nd.rate;
            }
            // 读突触
            int ssize = net.synapses.size();
            rfile.read((char *)&ssize, sizeof(int));
            for (int i = 0; i < ssize; i++)
            {
                synapse_data sd;
                rfile.read((char *)&sd, sizeof(synapse_data));
                net.synapses[i]=new Synapse();
                net.synapses[i]->src=sd.src;
                net.synapses[i]->tar=sd.tar;
                net.synapses[i]->delay=sd.delay;
                net.synapses[i]->weight=sd.weight;
            }
        }
        for(int i=0;i<net.synapses.size();i++){
            int src=net.synapses[i]->src;
            int tar=net.synapses[i]->tar;
            net.neurons[src]->adj.push_back(tar);
            net.neurons[tar]->adj.push_back(src);
        }
        rfile.close();
    };
};
