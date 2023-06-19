#include "util.h"

MGBrain::Nums::Nums(int _sum)
{
    sum = _sum;
    list = new int[sum];
}
MGBrain::Nums::~Nums()
{
    delete list;
}
int MGBrain::Nums::size()
{
    return sum;
}
int &MGBrain::Nums::operator[](int index)
{
    return list[index];
}