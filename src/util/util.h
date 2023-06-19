#pragma once
namespace MGBrain{
    class Nums{
    private:
        int* list;
        int sum;
    public:
        Nums(int _sum);
        ~Nums();
        int size();
        int& operator[](int index);
    };
    class ParameterParser{
        
    };
}