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
    time_t build_s,build_e,part_s,part_e,setup_s,setup_e,sim_s,sim_e;
    //设置时间片
    config::set_timestep(0.0001);
    config::enable_single_gpu();
    config::enable_fire_check();
    size_t nsyn=200'000'000;
    model m;
    float con_prob=0.02;
    int pop_size=static_cast<int>(std::sqrt((float)nsyn/(con_prob)));
    std::cout<<pop_size<<std::endl;
    maker::make_vogel(m,nsyn);
    ///构建网络
    build_s=clock();
    

    network net;
    netgen::gen_net(net,m);
    build_e=clock();
    std::cout<<"build:"<<(float)(build_e-build_s)/1000/1000<<" s"<<std::endl;
    
    ///划分网络
    int nparts=2;
    PartType ptype=PartType::MODEL;
    std::vector<int> part=parter::part_network(net,nparts,ptype);
    panalysis pana(part,net,nparts,ptype);
    pana.printAll();
    
    ///构建仿真
    mgsim sim(net,part,nparts);
    
    ///开始仿真
    sim.simulate(1);
    
    return 0;
}