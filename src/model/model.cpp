
#include "model.h"
MGBrain::Model::Model()
{
    indexer = 0;
}
void MGBrain::Model::set_lif_const(std::array<real, 30> _lifconst){
    nlifconst=true;
    lifconst=_lifconst;
}
void MGBrain::Model::set_stdp_const(std::array<real, 6> _stdpconst){
    nstdpconst=true;
    stdpconst=_stdpconst;
}
MGBrain::Population &MGBrain::Model::create_pop(int num, MGBrain::NeuronType type, bool isSource = false)
{
    pops.emplace_back(indexer++, num, isSource, type);
    return pops.back();
}
bool MGBrain::Model::connect(Population &src, Population tar, std::array<real, 2> _wrange, std::array<real, 2> _drange, float type)
{

    if (type == 0.0 && src.num != tar.num)
        return false;
    int index = -1;
    for (int i = 0; i < pros.size(); i++)
    {
        if (pros[i].src == src.id && (pros[i].tar) == tar.id)
        {
            index = i;
        }
    }
    if (index > 0)
    { // 覆盖
        pros[index].wrange = _wrange;
        pros[index].drange = _drange;
        pros[index].type = type;
    }
    else
    {
        pros.emplace_back(src.id, tar.id, _wrange, _drange, type);
    }
    return true;
}
bool MGBrain::Model::connect(Population &src, Population tar, real weight, real delay, float type)
{
    return connect(src, tar, {weight, weight}, {delay, delay}, type);
}