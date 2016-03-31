///////////////////////////////////////////////////////////
//
//  The C++ implementation of pairwise differences
//  
//  Important Nodes on Compilation !!!
//      - you must use mex -O to open the optimization
//        option, otherwise, the efficiency will be 
//        very low in run-time. (Due to the using of
//        template inline function in implementation
//        for core computation.)
//
//  History:
//      Created by Dahua Lin on Sep 10, 2006
//
///////////////////////////////////////////////////////////

#include "mex.h"

#include <math.h>

enum PwDiffMeasure
{
    pdmSum = 1,      // sum of all absolute differences
    pdmMax = 2,      // max of all absolute differences    
    pdmMin = 3,      // min of all absolute differences
};

template<PwDiffMeasure pdm> class DiffCollector;

template<> class DiffCollector<pdmSum>{
public: inline void calc(double& v0, double cv) { v0 += cv; } };

template<> class DiffCollector<pdmMax>{
public: inline void calc(double& v0, double cv) { if (v0 < cv) v0 = cv; } };

template<> class DiffCollector<pdmMin>{
public: inline void calc(double& v0, double cv) { if (v0 > cv) v0 = cv; } };

template<PwDiffMeasure pdm>
void calcpdm(const double* X1, const double* X2, double* D, int n1, int n2, int d);

void delegate_run(PwDiffMeasure pdmv, const double* X1, const double* X2, double* D, int n1, int n2, int d)
{
    switch (pdmv)
    {
        case pdmSum: calcpdm<pdmSum>(X1, X2, D, n1, n2, d); break;
        case pdmMax: calcpdm<pdmMax>(X1, X2, D, n1, n2, d); break;
        case pdmMin: calcpdm<pdmMin>(X1, X2, D, n1, n2, d); break;
    }
}


// core functions
// X1:  d x n1
// X2:  d x n2
// D:   n1 x n2

template<PwDiffMeasure pdm>
void calcpdm(const double* X1, const double* X2, double* D, int n1, int n2, int d)
{    
    double s = 0;
    double c = 0;    
    const double* curX1 = X1;
    const double* curX2 = X2;
    
    DiffCollector<pdm> collector;
    
    for (int i2 = 0; i2 < n2; ++i2)
    {              
        curX1 = X1;
        for (int i1 = 0; i1 < n1; ++i1)
        {                  
            s = fabs(*curX1 - *curX2);            
            for (int k = 1; k < d; ++k)
            {
                collector.calc(s, fabs(curX1[k] - curX2[k]));
            }
            *(D++) = s;
            
            curX1 += d;
        }
        curX2 += d;
    }
}

// Main Entry 
// Input:
//      X1, X2, pdmcode
// Output:
//      D
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // check input / output
    if (nrhs != 3)
        mexErrMsgTxt("The number of input should be 3.\n");
    if (nlhs > 1)
        mexErrMsgTxt("The number of output should not be larger than 1.\n");
    
    // get input
    const mxArray* mxX1 = prhs[0];
    const mxArray* mxX2 = prhs[1];
    const mxArray* mxpdm = prhs[2];
    int pdmi = (int)mxGetScalar(mxpdm);
    
    if (pdmi < 1 || pdmi > 3)
        mexErrMsgTxt("The pdm code should be from 1 to 3.\n");
    PwDiffMeasure pdm = (PwDiffMeasure)pdmi;
    
    // check sizes
    int d1 = mxGetM(mxX1);
    int n1 = mxGetN(mxX1);
    int d2 = mxGetM(mxX2);
    int n2 = mxGetN(mxX2);
    if (d1 != d2)
        mexErrMsgTxt("The sample dimensions of X1 and X2 are not the same.\n");
    
    // create destination matrix
    mxArray* mxD = mxCreateDoubleMatrix(n1, n2, mxREAL);
    
    // get base addresses
    double* X1 = mxGetPr(mxX1);
    double* X2 = mxGetPr(mxX2);
    double* D  = mxGetPr(mxD);
    
    // do computation
    delegate_run(pdm, X1, X2, D, n1, n2, d1);
    
    // assign output
    plhs[0] = mxD;    
}




