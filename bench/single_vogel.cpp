#include "../src/mgbrain.h"
#include <random>
#include <time.h>
#include <vector>
#include <iostream>
MGBrain::real random0001(){
    return rand() % (999 + 1) / (float)(999 + 1);
}
int main(int argc,char* argv[]){
    parser par(argc,argv);
    srand(time(0));
    time_t build_s,build_e,part_s,part_e,setup_s,setup_e,sim_s,sim_e;
    //设置时间片
    config::set_timestep(0.0001);
    size_t nsyn=20'000'000;
    float con_prob=0.02;
    int pop_size=static_cast<int>(std::sqrt((float)nsyn/(con_prob)));
    std::cout<<pop_size<<std::endl;
    ///构建网络
    build_s=clock();
    model m;
    pop pe=m.create_pop(pop_size*8/10,neuType::LIFE,false);
    pop pi=m.create_pop(pop_size*2/10,neuType::LIFE,false);
    real w1=0.4 * 16000000 / pop_size / pop_size;
    real w2=-5.1 * 16000000 / pop_size / pop_size;
    real d1=0.0008f;
    m.set_lif_const(constgen::gen_life_const1());
    m.connect(pe,pe,w1,d1,con_prob);
    m.connect(pe,pi,w1,d1,con_prob);
    m.connect(pi,pe,w2,d1,con_prob);
    m.connect(pi,pi,w2,d1,con_prob);
    network net;
    netgen::gen_net(net,m);
    build_e=clock();
    std::cout<<"build:"<<(float)(build_e-build_s)/1000/1000<<" s"<<std::endl;
    
    ///划分网络
    int nparts=2;
    PartType ptype=PartType::MODEL;
    std::vector<int> part=parter::part_network(net,nparts,ptype);
    
    ///构建仿真
    mgsim sim(net,part,nparts);
    
    ///开始仿真
    sim.simulate(1);
    
    return 0;
}