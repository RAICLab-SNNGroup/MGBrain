#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <time.h>
#include "../src/mgbrain.h"
MGBrain::real random0001(){
    return rand() % (999 + 1) / (float)(999 + 1);
}

int main(int argc,char* argv[])
{
    parser par(argc,argv);
    time_t build_s,build_e,part_s,part_e,setup_s,setup_e,sim_s,sim_e;
    //设置时间片
    config::set_timestep(0.0001);
<<<<<<< HEAD
    config::enable_single_gpu();
=======
    // config::enable_single_gpu();
>>>>>>> c4ab03fcdfabf7648c2c17bc67d4f44c60da2ec0
    config::enable_fire_check();
    ///构建网络
    build_s=clock();
    model m;
    size_t nsyn=par.getNum(parser::NSYN);
    float rate=par.getFloat(parser::RATE);
    maker::make_brunel(m,nsyn);

    network net;
    netgen::gen_net(net,m);
    build_e=clock();
    std::cout<<"build time:"<<(build_e-build_s)/1000<<"ms"<<std::endl;
    std::cout<<"neu size:"<<net.neurons.size()<<std::endl;
    std::cout<<"syn size:"<<net.synapses.size()<<std::endl;
    std::cout<<"rate:"<<rate<<std::endl;
    
    ///划分网络
    int npart=1;
    part_s=clock();
    std::vector<int> part=parter::part_network(net,npart,PartType::MODEL);
    part_e=clock();
    std::cout<<"part time:"<<(float)(part_e-part_s)/1000<<"ms"<<std::endl;

    ///设置初始参数
    for(int i=0;i<net.neurons.size();i++){
        if(net.neurons[i].type==NeuType::POISSON){
            net.neurons[i].rate=rate;
        }
    }
    
    ///构建仿真
    setup_s=clock();
    mgsim sim(net,part,npart);
    setup_e=clock();
    std::cout<<"steup time:"<<(setup_e-setup_s)/1000<<"ms"<<std::endl;
    
    ///开始仿真
    sim.simulate(1);
    std::cout<<"sim time:"<<sim.get_time()<<std::endl;
    return 0;
}
