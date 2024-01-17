#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <time.h>
#include "../src/mgbrain.h"
MGBrain::real random0001()
{
    return rand() % (999 + 1) / (float)(999 + 1);
}

int main()
{
    // 设置时间片
    config::set_timestep(0.0001);
    /// 构建网络
    model m;
    size_t nsyn = 250'000'000;
    maker::make_brunel(m,nsyn);

    network net;
    netgen::gen_net(net, m);
    std::cout << "neu size:" << net.neurons.size() << std::endl;
    std::cout << "syn size:" << net.synapses.size() << std::endl;

    /// 划分网络
    int npart = 2;
    std::vector<int> part = parter::part_network(net, npart, PartType::MODEL);
    panalysis pana("model", part, net, npart);
    pana.printAll();

    /// 设置初始参数
    for (int i = 0; i < net.neurons.size(); i++)
    {
        if (net.neurons[i].type == NeuType::POISSON)
        {
            net.neurons[i].rate = 0.002f;
        }
    }

    /// 构建仿真
    mgsim sim(net, part, npart);

    /// 开始仿真
    sim.simulate(1);
    return 0;
}
