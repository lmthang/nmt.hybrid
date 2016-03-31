///////////////////////////////////////////////////////////
//
//  The core of matrix-vector operation:
//      applying a vector to columns/rows of matrix
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

// The operation codes
enum OpCode
{
    opFill = 0,     // r = v
    opAdd  = 1,     // r = m + v
    opMul  = 2,     // r = m * v
    opMax  = 3,     // r = max(m, v)
    opMin  = 4      // r = min(m, v)
};

// The core functors to do operations

template<OpCode opc> class CoreFunctor;

template<> class CoreFunctor<opFill> {
public: inline double Calc(double m, double v){ return v; } };

template<> class CoreFunctor<opAdd> {
public: inline double Calc(double m, double v){ return m + v; } };

template<> class CoreFunctor<opMul> {
public: inline double Calc(double m, double v){ return m * v; } };

template<> class CoreFunctor<opMax> {
public: inline double Calc(double m, double v){ return (m > v) ? m : v; } };

template<> class CoreFunctor<opMin> {
public: inline double Calc(double m, double v){ return (m < v) ? m : v; } };


template<OpCode opc>
void apply_vecs(int d, const double* X, const double* v, double* Y, int m, int n);

// The function to delegate operations
void delegate_operations(OpCode opc, const double* X, const double* v, double* Y, int d, int m, int n)
{
    switch (opc)
        {
            case opFill: apply_vecs<opFill>(d, X, v, Y, m, n); break;
            case opAdd:  apply_vecs<opAdd>(d, X, v, Y, m, n); break;
            case opMul:  apply_vecs<opMul>(d, X, v, Y, m, n); break;
            case opMax:  apply_vecs<opMax>(d, X, v, Y, m, n); break;
            case opMin:  apply_vecs<opMin>(d, X, v, Y, m, n); break;
        }
}

// The template function to do whole calculations
template<OpCode opc>
void apply_vecs(int d, const double* X, const double* v, double* Y, int m, int n)
{
    CoreFunctor<opc> f;
    
    if (d == 1)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < m; ++i)
            {
                *(Y++) = f.Calc(*(X++), v[i]);
            }
        }
    }
    else
    {
        double cv = 0;
        for (int j = 0; j < n; ++j)
        {
            cv = v[j];
            for (int i = 0; i < m; ++i)
            {
                *(Y++) = f.Calc(*(X++), cv);
            }
        }
    }       
}




// The entry function
// Input:
//      X, v, d, opcode
// Output:
//      Y
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nlhs > 1)
    {
        mexErrMsgTxt("The number of outputs should not be larger than 1\n");
    }
    
    if (nrhs != 4)
    {
        mexErrMsgTxt("The number of inputs should be 4: X, v, d, opcode\n");
    }
    
    // get input
    const mxArray* mxX = prhs[0];
    const mxArray* mxv = prhs[1];
    const mxArray* mxd = prhs[2];
    const mxArray* mxopc = prhs[3];
    
    int m = mxGetM(mxX);
    int n = mxGetN(mxX);
    int vm = mxGetM(mxv);
    int vn = mxGetN(mxv);
    int d = (int)mxGetScalar(mxd);
    int opc = (int)mxGetScalar(mxopc);
    
    // check size
    if (d == 1)
    {
        if (vm != m || vn != 1)
            mexErrMsgTxt("The size of vector v is illegal\n");
    }
    else if (d == 2)
    {
        if (vm != 1 || vn != n)
            mexErrMsgTxt("The size of vector v is illegal\n");
    }
    else
    {
        mexErrMsgTxt("The value of d is illegal, it should be either 1 or 2\n");
    }
    
    // get base address
    const double* X = mxGetPr(mxX);
    const double* v = mxGetPr(mxv);
    
    mxArray* mxY = mxCreateDoubleMatrix(m, n, mxREAL);
    double* Y = mxGetPr(mxY);
    
    // do operation
    delegate_operations((OpCode)opc, X, v, Y, d, m, n);
    
    // make output
    plhs[0] = mxY;
    
}






