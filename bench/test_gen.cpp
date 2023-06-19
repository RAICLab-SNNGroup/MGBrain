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
    time_t build_s,build_e,part_s,part_e,setup_s,setup_e,sim_s,sim_e;
    //设置时间片
    config::set_timestep(0.0001);
    config::enable_single_gpu();
    config::enable_shuffle_gen();
    int nsyn=par.getNum(parser::NSYN);
    float con_prob=0.02;
    int pop_size=static_cast<int>(std::sqrt((float)nsyn/(con_prob)));
    std::cout<<pop_size<<std::endl;
    ///构建网络
    build_s=clock();
    model m;
    pop pe=m.create_pop(pop_size*8/10,neuType::LIF,false);
    pop pi=m.create_pop(pop_size*2/10,neuType::LIF,false);
    real w1=0.4 * 16000000 / pop_size / pop_size;
    real w2=-5.1 * 16000000 / pop_size / pop_size;
    real d1=0.0008f;
    m.connect(pe,pe,w1,d1,con_prob);
    m.connect(pe,pi,w1,d1,con_prob);
    m.connect(pi,pi,w2,d1,con_prob);
    m.connect(pi,pe,w2,d1,con_prob);
    network net;
    netgen::gen_net(net,m);
    build_e=clock();
    std::cout<<"build time:"<<(build_e-build_s)/1000<<"ms"<<std::endl;
    std::cout<<"neu size:"<<net.neurons.size()<<std::endl;
    std::cout<<"syn size:"<<net.synapses.size()<<std::endl;
    
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
    part_s=clock();
    std::vector<int> part=parter::part_network(net,nparts,ptype);
    part_e=clock();
    std::cout<<"part time:"<<(part_e-part_s)/1000<<"ms"<<std::endl;
    panalysis pana(typestr,part,net,nparts);
    pana.printAll();
    // int cost=pana.getCostNum();
    // float ratio=pana.getCostRatio();
    // float neusd=pana.getZoneNeu();
    // float synsd=pana.getZoneSyn();
    // float outsd=pana.getZoneOut();
    // float insd=pana.getZoneIn();
    // float wgtsd=pana.getZoneWgt();
    // // std::string dir=par.getStr(parser::PREFIX);
    // // std::string filename="opart_vogel.txt";
    // // std::ofstream ofs(dir+filename,std::ios::out|std::ios::app);
    // // if(ofs.is_open()){
    // //     ofs<<typestr<<","<<nparts<<","<<nsyn<<","<<cost<<","<<ratio<<","<<neusd<<","<<synsd<<","<<outsd<<","<<insd<<","<<wgtsd<<std::endl;
    // // }else{
    // //     std::cout<<"can't open file"<<std::endl;
    // // }
    return 0;
}