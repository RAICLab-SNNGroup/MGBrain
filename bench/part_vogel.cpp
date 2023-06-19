#include "../src/mgbrain.h"
#include <random>
#include <time.h>
#include <vector>
#include <iostream>
MGBrain::real random0001(){
    return rand() % (999 + 1) / (float)(999 + 1);
}
int main(int argc,char* argv[]){
    parser par(argc, argv);
    // 设置时间片
    config::set_timestep(0.0001);
    config::enable_sequence();
    /// 构建网络
    model m;
    float rate=par.getFloat(parser::RATE);
    size_t nsyn = par.getNum(parser::NSYN);
    int nparts = par.getNum(parser::NPART);
    std::string typestr = par.getStr(parser::MODEL);
    std::cout<<"============"<<std::endl;
    std::cout<<"model:vogel"<<std::endl;
    std::cout<<"nsyn:"<<nsyn<<std::endl;
    std::cout<<"nparts:"<<nparts<<std::endl;
    std::cout << "part method:" << typestr << std::endl;
    PartType ptype;
    
    if (typestr == "model")
    {
        ptype = PartType::MODEL;
    }
    else if (typestr == "metis")
    {
        ptype = PartType::METIS;
    }
    else if (typestr == "bsim")
    {
        ptype = PartType::BSIM;
    }
    else
    {
        ptype = PartType::SIMPLE;
    }
    maker::make_vogel(m, nsyn);
    network net;
    timer build;
    netgen::gen_net(net, m);
    std::cout << "build time:" << build.stop() <<" s"<< std::endl;

    /// 划分网络
    

    std::vector<int> part = parter::part_network(net, nparts, ptype);


    /// 构建仿真
    timer setup;
    mgsim sim(net, part, nparts);
    std::cout<<"setup:"<<setup.stop()<<" s"<<std::endl;
    /// 开始仿真
    
    sim.simulate(1);
    std::cout<<"sim time:"<<sim.get_time()<<" s"<<std::endl;
    std::cout<<"============"<<std::endl;
    return 0;
}