#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <time.h>
#include <fstream>
#include "../src/mgbrain.h"
int main(int argc,char* argv[])
{
    parser par(argc,argv);
    time_t build_s,build_e,part_s,part_e,setup_s,setup_e,sim_s,sim_e;
    //设置时间片
    config::set_timestep(0.0001);
    ///构建网络
    build_s=clock();
    model m;
    size_t nsyn=par.getNum(parser::NSYN);
    // long nsyn=1000;
    maker::make_brunel(m,nsyn);
    network net;
    netgen::gen_net(net,m);
    build_e=clock();
    
    ///划分网络
    int nparts=par.getNum(parser::NPART);
    PartType ptype;
    std::string typestr=par.getStr(parser::MODEL);
    if(typestr=="model"){
        ptype=PartType::MODEL;
    }else if(typestr=="metis"){
        ptype=PartType::METIS;
    }else if(typestr=="bsim"){
        ptype=PartType::BSIM;
    }else{
        ptype=PartType::SIMPLE;
    }
    std::vector<int> part=parter::part_network(net,nparts,ptype);

    panalysis pana(typestr,part,net,nparts);
    // int cost=pana.getCostNum();
    float ratio=pana.getCostRatio();
    // float neusd=pana.getZoneNeu();
    float synsd=pana.getZoneSyn();
    // float outsd=pana.getZoneOut();
    // float insd=pana.getZoneIn();
    float wgtsd=pana.getZoneWgt();
    std::string dir=par.getStr(parser::PREFIX);
    std::string filename="opart_brunel.txt";
    std::ofstream ofs(dir+filename,std::ios::out|std::ios::app);
    if(ofs.is_open()){
        ofs<<typestr<<","<<nparts<<","<<nsyn<<","<<ratio<<","<<synsd<<","<<wgtsd<<std::endl;
    }else{
        std::cout<<"can't open file"<<std::endl;
    }

    return 0;
}
