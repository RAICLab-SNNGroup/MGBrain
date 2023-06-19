#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <time.h>
#include "../src/mgbrain.h"
MGBrain::real random0001(){
    return rand() % (999 + 1) / (float)(999 + 1);
}

int main()
{
    //设置时间片
    config::set_timestep(0.0001);
    // config::enable_single_gpu();
    // config::enable_fire_check();
    config::enable_dense_spike();
    
    config::enable_staggered();
    ///构建网络
    model m;
    size_t nsyn=250'000'000;
    maker::make_brunel(m,nsyn);
    std::cout<<"model:brunel"<<std::endl;
    network net;
    timer build;
    netgen::gen_net(net,m);
    std::cout<<"build time:"<<build.stop()<<" s"<<std::endl;
    std::cout<<"neu size:"<<net.neurons.size()<<std::endl;
    std::cout<<"syn size:"<<net.synapses.size()<<std::endl;
    
    ///划分网络
    int npart=4;
    timer part_timer;
    std::cout<<"part:"<<npart<<std::endl;
    std::vector<int> part=parter::part_network(net,npart,PartType::MODEL);
    std::cout<<"part time:"<<part_timer.stop()<<" s"<<std::endl;
    // panalysis pana("model",part,net,npart);
    // pana.printAll();

    ///设置初始参数
    float rate=0.0001f;
    std::cout<<"rate:"<<rate<<std::endl;
    for(int i=0;i<net.neurons.size();i++){
        if(net.neurons[i].type==NeuType::POISSON){
            net.neurons[i].rate=rate;
        }
    }
    
    ///构建仿真
    timer setup;
    mgsim sim(net,part,npart);
    std::cout<<"steup time:"<<setup.stop()<<" s"<<std::endl;
    
    ///开始仿真
    sim.simulate(1);

    std::cout<<"sim_time:"<<sim.get_time()<<" s"<<std::endl;
    return 0;
}
