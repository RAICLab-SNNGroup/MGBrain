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
    time_t build_s,build_e,part_s,part_e,setup_s,setup_e,sim_s,sim_e;
    //设置时间片
    config::set_timestep(0.0001);
    config::enable_single_gpu();
    // config::enable_stdp();
    ///构建网络
    build_s=clock();
    model m;
    size_t nsyn=10'000'000;
    // long nsyn=1000;
    float con_prob=0.1;
    int pop_size=static_cast<int>(std::sqrt((float)nsyn/(con_prob*0.5)));
    // int pop_size=90;
    std::cout<<pop_size<<std::endl;
    m.set_lif_const(constgen::gen_life_const2());
    pop p=m.create_pop(pop_size/3,neuType::POISSON,true);
    pop e=m.create_pop(pop_size/3,neuType::LIF0,false);
    pop i=m.create_pop(pop_size/3,neuType::LIF0,false);
    real w1=0.0001 * 20000 / pop_size;
    real w2=-0.0005 * 20000 / pop_size;
    real d1=0.0014f;
    m.connect(p,e,w1,d1,con_prob);
    // m.connect(p,i,w1,d1,con_prob);
    // m.connect(e,e,w1,d1,con_prob);
    m.connect(e,i,w1,d1,con_prob);
    // m.connect(i,i,w2,d1,con_prob);
    // m.connect(i,e,w2,d1,con_prob);
    network net;
    netgen::gen_net(net,m);
    build_e=clock();
    std::cout<<"build time:"<<(build_e-build_s)/1000<<"ms"<<std::endl;
    std::cout<<"neu size:"<<net.neurons.size()<<std::endl;
    std::cout<<"syn size:"<<net.synapses.size()<<std::endl;
    
    ///划分网络
    int npart=1;
    part_s=clock();
    std::vector<int> part=parter::part_network(net,npart,PartType::MODEL);
    part_e=clock();
    std::cout<<"part time:"<<(float)(part_e-part_s)/1000<<"ms"<<std::endl;
    panalysis pana("model",part,net,npart);
    pana.printAll();

    ///设置初始参数
    for(int i=0;i<net.neurons.size();i++){
        if(net.neurons[i].type==NeuType::POISSON){
            net.neurons[i].rate=0.002f;
        }
    }
    
    ///构建仿真
    setup_s=clock();
    mgsim sim(net,part,npart);
    setup_e=clock();
    std::cout<<"steup time:"<<(setup_e-setup_s)/1000<<"ms"<<std::endl;
    
    ///开始仿真
    sim.simulate(1);
    return 0;
}
