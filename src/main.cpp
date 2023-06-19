#include <map>
#include <list>
#include <memory>
#include <algorithm>

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <time.h>

#include "mgbrain.h"

MGBrain::real random0001(){
    return rand() % (999 + 1) / (float)(999 + 1);
}

int main()
{
    time_t build_s,build_e,part_s,part_e,setup_s,setup_e,sim_s,sim_e;
    //设置时间片
    config::set_timestep(0.01);
    config::enable_single_gpu();
    ///构建网络
    build_s=clock();
    model m;
    int pop_size=1000;
    float con_prob=0.1;
    pop p0=m.create_pop(pop_size,neuType::POISSON,true);
    pop p1=m.create_pop(pop_size,neuType::POISSON,true);
    pop p2=m.create_pop(pop_size,neuType::LIF,false);
    pop p3=m.create_pop(pop_size,neuType::LIF,false);
    pop p4=m.create_pop(pop_size,neuType::LIF,false);
    m.connect(p1,p2,{1,2},{0.1,0.1},con_prob);
    m.connect(p2,p3,{1,2},{0.1,0.1},con_prob);
    m.connect(p0,p3,{1,2},{0.1,0.1},con_prob);
    m.connect(p3,p4,{1,2},{0.1,0.1},con_prob);
    network net;
    netgen::gen_net(net,m);
    build_e=clock();
    std::cout<<"build time:"<<(build_e-build_s)/1000<<"ms"<<std::endl;
    std::cout<<"neu size:"<<net.neurons.size()<<std::endl;
    std::cout<<"syn size:"<<net.synapses.size()<<std::endl;
    
    ///划分网络
    int npart=4;
    part_s=clock();
    std::vector<int> part=parter::part_network(net,npart,PartType::METIS);
    part_e=clock();
    std::cout<<"part time:"<<(part_e-part_s)/1000<<"ms"<<std::endl;
    panalysis pana("metis",part,net,npart);
    pana.printAll();
    ///设置初始参数
    for(int i=0;i<net.neurons.size();i++){
        if(net.neurons[i].type==NeuType::POISSON){
            net.neurons[i].rate=random0001();
        }
    }
    net.~Network();
    ///构建仿真
    setup_s=clock();
    // mgsim sim(net,part,npart);
    setup_e=clock();
    std::cout<<"steup time:"<<(setup_e-setup_s)/1000<<"ms"<<std::endl;
    
    ///开始仿真
    sim_s=clock();
    // sim.simulate(0.2);
    sim_e=clock();
    std::cout<<"sim time:"<<(sim_e-sim_s)/1000<<"ms"<<std::endl;

    
    return 0;
}
