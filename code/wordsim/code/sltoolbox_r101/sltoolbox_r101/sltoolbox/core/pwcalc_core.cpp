///////////////////////////////////////////////////////////
//
//  Pairwise Scalar Calculation Core
//      for a vector of length m and a vector of length n
//      compute an m x n matrix (calculation table)
//            
//  Important Nodes on Compilation !!!
//      - you must use mex -O to open the optimization
//        option, otherwise, the efficiency will be 
//        very low in run-time. (Due to the using of
//        template inline function in implementation
//        for core computation.)
//
//  History:
//    - Created by Dahua Lin, on Sep 10, 2006
//
///////////////////////////////////////////////////////////    


#include "mex.h"
#include <math.h>

enum OpCode
{
    opAdd     = 1,
    opMul     = 2,
    opAbsDiff = 3,
    opMax     = 4,
    opMin     = 5,
};

template<OpCode opc> class Calculator;

template<> class Calculator<opAdd>{
public: double calc(double a, double b) { return a + b; } };

template<> class Calculator<opMul>{
public: double calc(double a, double b) { return a * b; } };

template<> class Calculator<opAbsDiff>{
public: double calc(double a, double b) { return fabs(a - b); } };

template<> class Calculator<opMax>{
public: double calc(double a, double b) { return a > b ? a : b; } };

template<> class Calculator<opMin>{
public: double calc(double a, double b) { return a < b ? a : b; } };

template<OpCode opc>
void calc_matrix(const double* v1, const double* v2, double* M, int n1, int n2)
{
    Calculator<opc> C;
    for (int j = 0; j < n2; ++j)
    {
        double b = v2[j];
        for (int i = 0; i < n1; ++i)
        {
            *(M++) = C.calc(v1[i], b);
        }
    }    
}


void delegate_run(OpCode opc, const double* v1, const double* v2, double* M, int n1, int n2)
{
    switch (opc)
    {
        case opAdd:     calc_matrix<opAdd>(v1, v2, M, n1, n2); break;
        case opMul:     calc_matrix<opMul>(v1, v2, M, n1, n2); break;
        case opAbsDiff: calc_matrix<opAbsDiff>(v1, v2, M, n1, n2); break;
        case opMax:     calc_matrix<opMax>(v1, v2, M, n1, n2); break;
        case opMin:     calc_matrix<opMin>(v1, v2, M, n1, n2); break;
    }
}


// The entry function
// Input:
//      v1, v2, opcode
// Output:
//      M
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // check argument number
    if (nrhs != 3)
    {
        mexErrMsgTxt("The number of input arguments should be 3.\n");
    }
    if (nlhs > 1)
    {
        mexErrMsgTxt("The number of output arguments should not exceed 1.\n");
    }
    
    // get input
    const mxArray* mxv1 = prhs[0];
    const mxArray* mxv2 = prhs[1];
    const mxArray* mxopc = prhs[2];
    
    int opc = (int)mxGetScalar(mxopc);
    if (opc < 1 || opc > 5)
    {
        mexErrMsgTxt("The opcode should be between 1 and 5.\n");
    }
    
    // get sizes
    int n1 = mxGetNumberOfElements(mxv1);
    int n2 = mxGetNumberOfElements(mxv2);
    
    // create target matrix
    mxArray* mxM = mxCreateDoubleMatrix(n1, n2, mxREAL);
    
    // get base addresses
    const double* v1 = mxGetPr(mxv1);
    const double* v2 = mxGetPr(mxv2);
    double* M = mxGetPr(mxM);
    
    // do computation
    delegate_run((OpCode)opc, v1, v2, M, n1, n2);
    
    // assign output
    plhs[0] = mxM;
}





