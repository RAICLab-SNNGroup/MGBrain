#include <vector>
#include <iostream>
#include <string>
#include "../model/network.h"
#include "calculator.h"
namespace MGBrain
{
    class PartitionAnalysis
    {
        std::string name;
        std::vector<int> &part;
        Network &net;
        int nparts;

    public:
        PartitionAnalysis(std::string _name, std::vector<int> &_part, Network &_net, int _nparts) : name(_name), part(_part), net(_net), nparts(_nparts) {}
        PartitionAnalysis(std::vector<int> &_part, Network &_net, int _nparts,PartType type) : part(_part), net(_net), nparts(_nparts)
        {
            if(type==PartType::BSIM){
                name="bsim";
            }else if(type==PartType::METIS){
                name="metis";
            }else if(type==PartType::MODEL){
                name="model";
            }else{
                name="simple";
            }
        }
        int getCostNum(bool print = false);
        float getCostRatio(bool print = false);
        float getZoneNeu(bool print = false);
        float getZoneWgt(bool print = false);
        float getZoneSyn(bool print = false);
        float getZoneOut(bool print = false);
        float getZoneIn(bool print = false);
        void printAll();
    };
};