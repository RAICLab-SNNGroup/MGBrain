#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <time.h>
#include <unordered_set>
#include "../src/mgbrain.h"
MGBrain::real random0001(){
    return rand() % (999 + 1) / (float)(999 + 1);
}

int main()
{
    model m;
    size_t nsyn=250'000'000;
    maker::make_brunel(m,nsyn);
    std::cout<<"model:brunel"<<std::endl;
    network net;
    timer build;
    netgen::gen_net(net,m);
    std::cout<<"build time:"<<build.stop()<<" s"<<std::endl;
    std::cout<<"neu size:"<<net.neurons.size()<<std::endl;
    std::cout<<"syn size:"<<net.synapses.size()<<std::endl;
    
    ///划分网络
    int npart=8;
    timer part_timer;
    std::cout<<"part:"<<npart<<std::endl;
    std::vector<int> part=parter::part_network(net,npart,PartType::BSIM);
    std::cout<<"part time:"<<part_timer.stop()<<" s"<<std::endl;
    panalysis pana("model",part,net,npart);
    pana.printAll();
    std::unordered_set<int> set;
    for(int i=0;i<part.size();i++){
        set.insert(part[i]);
    }
    std::cout<<"parts:"<<set.size()<<std::endl;
    return 0;
}