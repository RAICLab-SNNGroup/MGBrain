#include "metisutil.h"

int MGBrain::MetisUtil::metis_part_normal_graph(idx_t nCells, int nPart, std::vector<idx_t> &xadj, std::vector<idx_t> &adjncy, std::vector<idx_t> &part)
{
    idx_t ncon = 1;
    idx_t *vwgt = 0;
    idx_t *vsize = 0;
    idx_t *adjwgt = 0;
    real_t *tpwgts = 0;
    real_t *ubvec = 0;
    idx_t options[METIS_NOPTIONS];
    idx_t wgtflag = 0;
    idx_t numflag = 0;
    idx_t objval;
    idx_t nZone = nPart;
    idx_t *xadjp = xadj.data();
    idx_t *adjncyp = adjncy.data();
    idx_t *cellzonep = part.data();

    METIS_SetDefaultOptions(options);
    if (nZone > 8)
    {
        METIS_PartGraphKway(&nCells, &ncon, xadjp, adjncyp, vwgt, vsize, adjwgt,
                            &nZone, tpwgts, ubvec, options, &objval, cellzonep);
    }
    else
    {
        METIS_PartGraphRecursive(&nCells, &ncon, xadjp, adjncyp, vwgt, vsize, adjwgt,
                                 &nZone, tpwgts, ubvec, options, &objval, cellzonep);
    }
    return objval;
}
int MGBrain::MetisUtil::metis_part_weighted_graph(int nvtxs, int nparts,  std::vector<idx_t> &xadj, std::vector<idx_t> &adjncy, std::vector<idx_t> &vwgt,
                                                  std::vector<idx_t> &adjwgt, std::vector<idx_t> &part)
{
    idx_t options[METIS_NOPTIONS];
    idx_t objval;
    idx_t ncell = nvtxs;
    idx_t nzone = nparts;
    idx_t ncons = 1;
    std::vector<real_t>tpwgts;
    std::vector<real_t>ubvec;
    std::vector<idx_t> vsize;
    vsize.resize(nvtxs,1);
    METIS_SetDefaultOptions(options);
    if (nzone > 8)
    {
        METIS_PartGraphKway(&ncell, &ncons, xadj.data(), adjncy.data(), vwgt.data(), vsize.data(), adjwgt.data(),
                            &nzone, tpwgts.data(), ubvec.data(), options, &objval, part.data());
    }
    else
    {
        METIS_PartGraphRecursive(&ncell, &ncons, xadj.data(), adjncy.data(), vwgt.data(), vsize.data(), adjwgt.data(),
                                 &nzone, tpwgts.data(), ubvec.data(), options, &objval, part.data());
    }
    return objval;
}
