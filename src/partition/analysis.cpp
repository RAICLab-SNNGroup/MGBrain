#include "analysis.h"

int MGBrain::PartitionAnalysis::getCostNum(bool print)
{
    int cost = 0;
    for (int i = 0; i < net.synapses.size(); i++)
    {
        int src = net.synapses[i].src;
        int tar = net.synapses[i].tar;
        if (part[src] != part[tar])
        {
            cost++;
        }
    }
    if (print)
        std::cout << name << " part cost:" << cost << std::endl;
    return cost;
}
float MGBrain::PartitionAnalysis::getCostRatio(bool print)
{
    int cost = getCostNum();
    float rate = (float)cost / (float)net.synapses.size();
    if (print)
        std::cout << name << " cut-rate:" << rate << "(" << cost << "/" << net.synapses.size() << ")" << std::endl;
    return rate;
}
float MGBrain::PartitionAnalysis::getZoneNeu(bool print)
{
    std::vector<int> zones(nparts, 0);
    float avg = (float)net.neurons.size() / nparts;

    for (int i = 0; i < net.neurons.size(); i++)
    {
        zones[part[i]]++;
    }
    if (print)
        std::cout << name << " part zone neu:";
    float sum = 0;
    for (int i = 0; i < zones.size(); i++)
    {
        if (print)
            std::cout << "(" << i << ":" << zones[i] << ")";
        sum += (zones[i] - avg) * (zones[i] - avg);
    }
    float sd = std::sqrt(sum / nparts);
    if (print)
        std::cout << "\t Standard Deviation:" << sd << std::endl;
    return sd;
}
float MGBrain::PartitionAnalysis::getZoneWgt(bool print)
{
    std::vector<int> zones(nparts, 0);
    int wgtsum = 0;
    for (int i = 0; i < net.neurons.size(); i++)
    {
        float edgewgts = 0;
        for (int j = 0; j < net.neurons[i].nxt.size(); j++)
        {
            edgewgts += ModelCalculator::synapseWeight(SynapseType::STATIC, 0.01);
        }
        int neuwgt = ModelCalculator::neuronWeight(net.neurons[i].type, 0.5, edgewgts);
        zones[part[i]] += neuwgt;
        wgtsum += neuwgt;
    }
    float wgtavg = (float)wgtsum / nparts;
    if (print)
        std::cout << name << " part zone wgt:";
    float sum = 0;
    for (int i = 0; i < zones.size(); i++)
    {
        if (print)
            std::cout << "(" << i << ":" << zones[i] << ")";
        sum += (zones[i] - wgtavg) * (zones[i] - wgtavg);
    }
    float sd = std::sqrt(sum / nparts);
    if (print)
        std::cout << "\t Standard Deviation:" << sd << std::endl;
    return sd;
}
float MGBrain::PartitionAnalysis::getZoneSyn(bool print)
{
    std::vector<int> zones(nparts, 0);
    float synavg = (float)net.synapses.size() / nparts;
    for (int i = 0; i < net.synapses.size(); i++)
    {
        int src = net.synapses[i].src;
        zones[part[src]]++;
    }
    if (print)
        std::cout << name << " part zone syn:";
    float sum = 0;
    for (int i = 0; i < zones.size(); i++)
    {
        if (print)
            std::cout << "(" << i << ":" << zones[i] << ")";
        sum += (zones[i] - synavg) * (zones[i] - synavg);
    }
    float sd = std::sqrt(sum / nparts);
    if (print)
        std::cout << "\t Standard Deviation:" << sd << std::endl;
    return sd;
}
float MGBrain::PartitionAnalysis::getZoneOut(bool print)
{
    std::vector<int> zones(nparts, 0);
    int outsum = 0;
    for (int i = 0; i < net.synapses.size(); i++)
    {
        int src = net.synapses[i].src;
        int tar = net.synapses[i].tar;
        if (part[src] != part[tar])
        {
            zones[part[src]]++;
            outsum++;
        }
    }
    float outavg = (float)outsum / nparts;
    if (print)
        std::cout << name << " part zone outter:";
    float sum = 0;
    for (int i = 0; i < zones.size(); i++)
    {
        if (print)
            std::cout << "(" << i << ":" << (zones[i]) << ")";
        sum += (zones[i] - outavg) * (zones[i] - outavg);
    }
    float sd = std::sqrt(sum / nparts);
    if (print)
        std::cout << "\t Standard Deviation:" << sd << std::endl;
    return sd;
}
float MGBrain::PartitionAnalysis::getZoneIn(bool print)
{
    std::vector<int> zones(nparts, 0);
    int insum = 0;
    for (int i = 0; i < net.synapses.size(); i++)
    {
        int src = net.synapses[i].src;
        int tar = net.synapses[i].tar;
        if (part[src] == part[tar])
        {
            zones[part[src]]++;
            insum++;
        }
    }
    float inavg = (float)insum / nparts;
    if (print)
        std::cout << name << " part zone inner:";
    float sum = 0;
    for (int i = 0; i < zones.size(); i++)
    {
        if (print)
            std::cout << "(" << i << ":" << (zones[i]) << ")";
        sum += (zones[i] - inavg) * (zones[i] - inavg);
    }
    float sd = std::sqrt(sum / nparts);
    if (print)
        std::cout << "\t Standard Deviation:" << sd << std::endl;
    return sd;
}
void MGBrain::PartitionAnalysis::printAll()
{
    getCostNum(true);
    getCostRatio(true);
    getZoneNeu(true);
    getZoneSyn(true);
    getZoneOut(true);
    getZoneIn(true);
    getZoneWgt(true);
    
}