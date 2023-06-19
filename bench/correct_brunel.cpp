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
    time_t build_s, build_e, part_s, part_e, setup_s, setup_e, sim_s, sim_e;
    // 设置时间片
    config::set_timestep(0.0001);
    // config::enable_dense_spike();
    // config::enable_staggered();
    config::enable_fire_check();
    /// 构建网络
    build_s = clock();
    model m;
    size_t nsyn = 250'000'000;
    maker::make_brunel(m,nsyn);

    network net;
    netgen::gen_net(net, m);
    build_e = clock();
    std::cout << "build time:" << (build_e - build_s) / 1000 << "ms" << std::endl;
    std::cout << "neu size:" << net.neurons.size() << std::endl;
    std::cout << "syn size:" << net.synapses.size() << std::endl;

    /// 划分网络
    int npart = 8;
    part_s = clock();
    std::vector<int> part = parter::part_network(net, npart, PartType::MODEL);
    part_e = clock();
    std::cout << "part time:" << (float)(part_e - part_s) / 1000 << "ms" << std::endl;
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
    setup_s = clock();
    mgsim sim(net, part, npart);
    setup_e = clock();
    std::cout << "steup time:" << (setup_e - setup_s) / 1000 << "ms" << std::endl;

    /// 开始仿真
    sim.simulate(1);
    return 0;
}
