#pragma once
#include "../../struct.h"
#include <omp.h>
namespace MGBrain{
    struct OffsetMap
    {
        /* data */
        int size;
        int* offset;
    };
    struct ZoneMap
    {
        /* data */
        int size;
        short* zone;
    };
    struct NeusList{
        NEUBlock block;
    };
    
    
};