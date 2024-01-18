#include "netgen.h"
void MGBrain::NetGen::rand_vector(std::vector<bool> &list, int gen_num, int max_num)
{
    list.resize(max_num);
    // srand((int)time(0));
    for (size_t i = 0; i < max_num; i++)
    {
        if (i < gen_num)
            list[i] = true;
        else
            list[i] = false;
    }
    int index = max_num;
    while (--index)
    {
        std::swap(list[index], list[rand() % max_num]);
    }
}
MGBrain::real MGBrain::NetGen::rand_weight(real floor, real ceil)
{
    if (ceil == floor)
        return floor;

    int range = (int)((ceil - floor) / Config::DW) + 1;
    real tw;
    if (floor < 0)
        tw = floor + (rand() % range) * Config::DW;
    else
        tw = floor - (rand() % range) * Config::DW;
    return tw;
}
MGBrain::real MGBrain::NetGen::rand_delay(real floor, real ceil)
{
    if (ceil == floor)
        return floor;
    int range = (int)((ceil - floor) / Config::DT) + 1;
    return floor + (rand() % range) * Config::DT;
}
void MGBrain::NetGen::gen_net(Network &net, Model &model)
{
    // if(Config::SHUFFLE_GEN){
    //     std::cout<<"gen net with std::random_shuffle"<<std::endl;
    // }else{
    //     std::cout<<"gen net without std::random_shuffle"<<std::endl;
    // }
    // time_t neu_s,neu_e;
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    net.nlifconst=model.nlifconst;
    net.nstdpconst=model.nstdpconst;
    net.lifconst=model.lifconst;
    net.stdpconst=model.stdpconst;

    net.pops.reserve(model.pops.size());
    // 构建族群和神经元
    //  neu_s=clock();
    int nindexer = 0;
    for (int i = 0; i < model.pops.size(); i++)
    {
        auto &pop = model.pops[i];
        net.pops.emplace_back(pop.id, pop.num, pop.source, pop.type);
        net.neurons.reserve(net.neurons.size() + pop.num);
        for (int j = 0; j < pop.num; j++)
        {
            int id = nindexer++;
            net.pops[i].neurons[j] = id;
            net.neurons.emplace_back(id, pop.source, pop.type);
        }
    }
    model.indexer=nindexer;
    // neu_e=clock();
    // float neu_time = (float)(neu_e - neu_s) / 1000 / 1000;
    // std::cout << "neu time:" << neu_time << " s" << std::endl;
    // 构建突触
    for (int i = 0; i < model.pros.size(); i++)
    {
        int srcpop = model.pros[i].src;
        int tarpop = model.pros[i].tar;
        int sn = model.pops[srcpop].num;
        int tn = model.pops[tarpop].num;
        real minw = model.pros[i].wrange[0];
        real maxw = model.pros[i].wrange[1];
        real mind = model.pros[i].drange[0];
        real maxd = model.pros[i].drange[1];
        if (model.pros[i].ctype == 1.0)
        {
            net.synapses.reserve(net.synapses.size() + sn * tn);
            for (int m = 0; m < sn; m++)
            {
                int src = net.pops[srcpop].neurons[m];
                net.neurons[src].nxt.reserve(net.neurons[src].nxt.size()+tn);
            }
            for (int n = 0; n < tn; n++)
            {
                int tar = net.pops[tarpop].neurons[n];
                net.neurons[tar].pre.reserve(net.neurons[tar].nxt.size()+sn);
            }
            for (int m = 0; m < sn; m++)
            {
                for (int n = 0; n < tn; n++)
                {
                    int src = net.pops[srcpop].neurons[m];
                    int tar = net.pops[tarpop].neurons[n];
                    real wgt = rand_weight(minw, maxw);
                    real delay = rand_delay(mind, maxd);
                    net.synapses.emplace_back(src, tar, wgt, delay);
                    net.neurons[src].nxt.push_back(tar);
                    net.neurons[tar].pre.push_back(src);
                    net.neurons[src].syns.push_back(net.synapses.size());
                }
            }
        }
        else if (model.pros[i].ctype == 0.0)
        {
            net.synapses.reserve(net.synapses.size() + sn);
            for (int k = 0; k < sn; k++)
            {
                int src = net.pops[srcpop].neurons[k];
                int tar = net.pops[tarpop].neurons[k];
                real wgt = rand_weight(minw, maxw);
                real delay = rand_delay(mind, maxd);
                net.synapses.emplace_back(src, tar, wgt, delay);
                net.neurons[src].nxt.push_back(tar);
                net.neurons[tar].pre.push_back(src);
                net.neurons[src].syns.push_back(net.synapses.size());
            }
        }
        else if (model.pros[i].ctype < 1.0 && model.pros[i].ctype > 0.0)
        {
            if (Config::SHUFFLE_GEN)
            {
                std::vector<bool> genlist(sn * tn);
                size_t gen_num = std::round(genlist.size() * model.pros[i].ctype);
                for (int k = 0; k < genlist.size(); k++)
                {
                    if (k < gen_num)
                    {
                        genlist[k] = true;
                    }
                    else
                    {
                        genlist[k] = false;
                    }
                }
                std::random_shuffle(genlist.begin(), genlist.end());
                net.synapses.reserve(net.synapses.size() + gen_num);
                for (int m = 0; m < sn; m++)
                {
                    for (int n = 0; n < tn; n++)
                    {
                        if (genlist[m * tn + n])
                        {
                            int src = net.pops[srcpop].neurons[m];
                            int tar = net.pops[tarpop].neurons[n];
                            real wgt = rand_weight(minw, maxw);
                            real delay = rand_delay(mind, maxd);
                            net.synapses.emplace_back(src, tar, wgt, delay);
                            net.neurons[src].nxt.push_back(tar);
                            net.neurons[tar].pre.push_back(src);
                            net.neurons[src].syns.push_back(net.synapses.size());
                        }
                    }
                }
            }
            else
            {
                /// 避免溢出
                // time_t gen_s,gen_e,span_s,span_e,syn_s,syn_e;
                // gen_s=clock();
                size_t max_num = (size_t)sn * (size_t)tn;
                size_t gen_num = static_cast<size_t>(std::round(max_num * model.pros[i].ctype));
                size_t count = 0;
                std::vector<size_t> genlist(gen_num);
                std::vector<size_t> snxtn(sn);
                std::vector<size_t> tpren(tn);
                // 生成随机突触序列
                for (size_t m = 0; m < sn; m++)
                {
                    for (size_t n = 0; n < tn; n++)
                    {
                        if (count < gen_num && ((rng() % max_num) < (gen_num * 1.1)))
                        {
                            genlist[count++] = m * tn + n;
                            snxtn[m]++;
                            tpren[n]++;
                        }
                    }
                }
                // gen_e=clock();
                // float gen_time = (float)(gen_e - gen_s) / 1000 / 1000;
                // std::cout << i <<" gen time:" << gen_time << " s" << std::endl;
                // span_s=clock();

                // 分配空间
                for (int m = 0; m < sn; m++)
                {
                    int srcn = net.pops[srcpop].neurons[m];
                    net.neurons[srcn].nxt.reserve(net.neurons[srcn].nxt.size() + snxtn[m]);
                }
                for (int n = 0; n < tn; n++)
                {
                    int tarn = net.pops[tarpop].neurons[n];
                    net.neurons[tarn].pre.reserve(net.neurons[tarn].pre.size() + tpren[n]);
                }
                net.synapses.reserve(net.synapses.size() + gen_num);
                // span_e=clock();
                // float span_time = (float)(span_e - span_s) / 1000 / 1000;
                // std::cout << i <<" span time:" << span_time << " s" << std::endl;
                // syn_s=clock();
                // 按照随机生成序列生成突触数据
                for (size_t t = 0; t < genlist.size(); t++)
                {
                    int m = genlist[t] / tn;
                    int n = genlist[t] % tn;
                    int src = net.pops[srcpop].neurons[m];
                    int tar = net.pops[tarpop].neurons[n];
                    real wgt = rand_weight(minw, maxw);
                    real delay = rand_delay(mind, maxd);
                    net.synapses.emplace_back(src, tar, wgt, delay);
                    net.neurons[src].nxt.push_back(tar);
                    net.neurons[tar].pre.push_back(src);
                    net.neurons[src].syns.push_back(net.synapses.size());
                }
                // syn_e=clock();
                // float syn_time = (float)(syn_e - syn_s) / 1000 / 1000;
                // std::cout << i <<" syn time:" << syn_time << " s" << std::endl;
            }
        }
    }
}
// void MGBrain::Network::push_pop(Population *p)
//     pops.push_back(p);
// int MGBrain::Network::push_neuron(NeuronType type, bool isSource)
//     int id = indexer++;
//     Neuron *n=new Neuron();
//     n->id=id;
//     n->source=isSource;
//     n->type=type;
//     n->rate=0;
//     neurons.push_back(n);
//     return n->id;
// void MGBrain::Network::push_synapse(int src, int tar, real weight, real delay)
//     neurons[src]->adj.push_back(tar);
//     neurons[tar]->adj.push_back(src);
//     neurons[src]->nxt.push_back(tar);
//     auto *syn = new Synapse();
//     syn->tar = tar;
//     syn->src = src;
//     syn->weight = weight;
//     syn->delay = delay;
//     synapses.push_back(syn);

// MGBrain::Network &MGBrain::NetGen::gen_net_mp(Model *model)
// {
//     auto *net = new Network(model);
//     int nindexer = 0;
//     int sindexer = 0;
//     int pop_size = model->pops.size();
//     int neu_size = 0;
//     int syn_size = 0;
//     for (int i = 0; i < pop_size; i++)
//     {
//         neu_size += model->pops[i]->num;
//     }
//     for (int i = 0; i < model->pros.size(); i++)
//     {
//         int sn = model->pops[model->pros[i]->src]->num;
//         int tn = model->pops[model->pros[i]->tar]->num;
//         float type = model->pros[i]->type;
//         if (type == 1.0)
//         {
//             syn_size += sn * tn;
//         }
//         else if (type == 0.0)
//         {
//             syn_size += sn;
//         }
//         else if (type < 1.0 && type > 0.0)
//         {
//             syn_size += std::round(sn * tn * type);
//         }
//     }
//     net.pops.resize(pop_size);
//     net.neurons.resize(neu_size);
//     net.synapses.resize(syn_size);
//     for (int i = 0; i < model->pops.size(); i++)
//     {
//         int pid = model->pops[i]->id;
//         int num = model->pops[i]->num;
//         auto type = model->pops[i]->type;
//         auto source = model->pops[i]->source;
//         net.pops[i] = model->pops[i];
// #pragma omp parallel for
//         for (int j = 0; j < num; j++)
//         {
//             int id = 0;
// #pragma omp critical
//             {
//                 id = nindexer++;
//                 net.pops[i]->neurons.push_back(id);
//             }
//             auto n = new Neuron();
//             n->id = id;
//             n->source = source;
//             n->type = type;
//             n->rate = 0;
//             net.neurons[id] = n;
//         }
//     }

//     for (int i = 0; i < model->pros.size(); i++)
//     {
//         int srcpop = model->pros[i]->src;
//         int tarpop = model->pros[i]->tar;
//         int sn = model->pops[srcpop]->num;
//         int tn = model->pops[tarpop]->num;
//         real minw = model->pros[i]->wrange[0];
//         real maxw = model->pros[i]->wrange[1];
//         real mind = model->pros[i]->drange[0];
//         real maxd = model->pros[i]->drange[1];
//         if (model->pros[i]->type == 1.0) // 全连接
//         {
// #pragma omp parallel for
//             for (int k = 0; k < sn * tn; k++)
//             {
//                 int m = k / tn;
//                 int n = k % tn;
//                 real randwgt = rand_weight(minw, maxw);
//                 real randdly = rand_delay(mind, maxd);
//                 int src = model->pops[srcpop]->neurons[m];
//                 int tar = model->pops[tarpop]->neurons[n];
//                 int ref = 0;
// #pragma omp critical
//                 {
//                     ref = sindexer++;
//                     net.neurons[src]->adj.push_back(tar);
//                     net.neurons[tar]->adj.push_back(src);
//                     net.neurons[src]->nxts++;
//                 }
//                 auto *syn = new Synapse();
//                 syn->tar = tar;
//                 syn->src = src;
//                 syn->weight = randwgt;
//                 syn->delay = randdly;
//                 net.synapses[ref] = syn;
//             }
//         }
//         else if (model->pros[i]->type == 0.0)
//         {
// #pragma omp parallel for
//             for (int k = 0; k < sn; k++)
//             {
//                 real randwgt = rand_weight(minw, maxw);
//                 real randdly = rand_delay(mind, maxd);
//                 int src = model->pops[srcpop]->neurons[k];
//                 int tar = model->pops[tarpop]->neurons[k];
//                 int ref = 0;
// #pragma omp critical
//                 {
//                     ref = sindexer++;
//                     net.neurons[src]->adj.push_back(tar);
//                     net.neurons[tar]->adj.push_back(src);
//                     net.neurons[src]->nxts++;
//                 }
//                 auto *syn = new Synapse();
//                 syn->tar = tar;
//                 syn->src = src;
//                 syn->weight = randwgt;
//                 syn->delay = randdly;
//                 net.synapses[ref] = syn;
//             }
//         }
//         else if (model->pros[i]->type < 1.0 && model->pros[i]->type > 0.0)
//         {
//             std::vector<bool> genlist(sn * tn);
//             for (int k = 0; k < genlist.size(); k++)
//             {
//                 if (k < std::round(sn * tn * model->pros[i]->type))
//                     genlist[k] = true;
//                 else
//                     genlist[k] = false;
//             }
//             std::random_shuffle(genlist.begin(), genlist.end());
// #pragma omp parallel for
//             for (int k = 0; k < genlist.size(); k++)
//             {
//                 if (genlist[k])
//                 {
//                     int m = k / tn;
//                     int n = k % tn;
//                     real randwgt = rand_weight(minw, maxw);
//                     real randdly = rand_delay(mind, maxd);
//                     int src = model->pops[srcpop]->neurons[m];
//                     int tar = model->pops[tarpop]->neurons[n];
//                     int ref = 0;
// #pragma omp critical
//                     {
//                         ref = sindexer++;
//                         net.neurons[src]->adj.push_back(tar);
//                         net.neurons[tar]->adj.push_back(src);
//                         net.neurons[src]->nxts++;
//                     }
//                     auto *syn = new Synapse();
//                     syn->tar = tar;
//                     syn->src = src;
//                     syn->weight = randwgt;
//                     syn->delay = randdly;
//                     net.synapses[ref] = syn;
//                 }
//             }
//         }
//     }
//     return *net;
// }