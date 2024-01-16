
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
bool MGBrain::Model::connect(Population &src, Population tar, std::array<real, 2> _wrange, std::array<real, 2> _drange, float _ctype,SynapseType _stype)
{

    if (_ctype == 0.0 && src.num != tar.num)
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
        pros[index].ctype = _ctype;
        pros[index].stype = _stype;
    }
    else
    {
        pros.emplace_back(src.id, tar.id, _wrange, _drange, _ctype,_stype);
    }
    return true;
}
bool MGBrain::Model::connect(Population &src, Population tar, real weight, real delay, float ctype,SynapseType stype)
{
    return connect(src, tar, {weight, weight}, {delay, delay}, ctype,stype);
}
bool MGBrain::Model::connect(Population &src, Population tar, std::array<real, 2> _wrange, std::array<real, 2> _drange, float ctype)
{
    return connect(src, tar, _wrange,_drange, ctype,SynapseType::STATIC);
}
bool MGBrain::Model::connect(Population &src, Population tar, real weight, real delay, float ctype)
{
    return connect(src, tar, {weight, weight}, {delay, delay}, ctype,SynapseType::STATIC);
}