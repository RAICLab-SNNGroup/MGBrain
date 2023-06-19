#include "../src/mgbrain.h"
#include <random>
#include <time.h>
#include <vector>
#include <iostream>
MGBrain::real random0001(){
    return rand() % (999 + 1) / (float)(999 + 1);
}
int main(int argc,char* argv[]){
    srand(time(0));
    //设置时间片
    config::set_timestep(0.0001);
    config::enable_fire_check();
    config::enable_dense_spike();
    // config::enable_sequence();
    size_t nsyn=800'000'000;
    model m;
    maker::make_vogel(m,nsyn);
    network net;
    timer build;
    netgen::gen_net(net,m);
    std::cout<<"build:"<<build.stop()<<" s"<<std::endl;
    
    ///划分网络
    int nparts=8;
    PartType ptype=PartType::MODEL;
    std::vector<int> part=parter::part_network(net,nparts,ptype);
    panalysis pana(part,net,nparts,ptype);
    pana.printAll();
    
    ///构建仿真
    timer setup;
    mgsim sim(net,part,nparts);
    std::cout<<"setup time:"<<setup.stop()<<" s"<<std::endl;
    ///开始仿真
    sim.simulate(1);

    std::cout<<"sim_time:"<<sim.get_time()<<" s"<<std::endl;
    
    return 0;
}