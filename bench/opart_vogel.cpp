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
    //设置时间片
    config::set_timestep(0.0001);
    model m;
    size_t nsyn=par.getNum(parser::NSYN);
    maker::make_vogel(m,nsyn);
    network net;
    netgen::gen_net(net,m);
    
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
    std::string filename="opart_vogel2.txt";
    std::ofstream ofs(dir+filename,std::ios::out|std::ios::app);
    if(ofs.is_open()){
        ofs<<typestr<<","<<nparts<<","<<nsyn<<","<<ratio<<","<<synsd<<","<<wgtsd<<std::endl;
    }else{
        std::cout<<"can't open file"<<std::endl;
    }
    return 0;
}