///////////////////////////////////////////////////////////
//
//  The core of matrix-vector operation:
//      simultaneously applying a column vector to 
//      every columns and a row vector to every rows
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
    opAdd  = 1,     // r = m + v
    opMul  = 2,     // r = m * v
};

// The core functors to do operations

template<OpCode opc> class CoreFunctor;


template<> class CoreFunctor<opAdd> {
public: inline double Calc(double m, double vr, double vc){ return m + vr + vc; } };

template<> class CoreFunctor<opMul> {
public: inline double Calc(double m, double vr, double vc){ return m * vr * vc; } };


template<OpCode opc>
void apply_rowcols(const double* X, const double* vrow, const double* vcol, double* Y, int m, int n);

// The function to delegate operations
void delegate_operations(int opc, const double* X, const double* vrow, const double* vcol, double* Y, int m, int n)
{
    switch (opc)
        {
            case opAdd:  apply_rowcols<opAdd>(X, vrow, vcol, Y, m, n); break;
            case opMul:  apply_rowcols<opMul>(X, vrow, vcol, Y, m, n); break;
        }
}

// The template function to do whole calculations
template<OpCode opc>
void apply_rowcols(const double* X, const double* vrow, const double* vcol, double* Y, int m, int n)
{
    CoreFunctor<opc> f;
    
    double cvr = 0;
    for (int j = 0; j < n; ++j)
    {
        cvr = vrow[j];
        for (int i = 0; i < m; ++i)
        {
            *(Y++) = f.Calc(*(X++), cvr, vcol[i]);
        }
    }        
}


// The entry function
// Input:
//      X, vrow, vcol, opcode
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
        mexErrMsgTxt("The number of inputs should be 4: X, vrow, vcol, opcode\n");
    }
    
    // get input
    const mxArray* mxX = prhs[0];
    const mxArray* mxvrow = prhs[1];
    const mxArray* mxvcol = prhs[2];
    const mxArray* mxopc = prhs[3];
    
    int m = mxGetM(mxX);
    int n = mxGetN(mxX);
    int vrm = mxGetM(mxvrow);
    int vrn = mxGetN(mxvrow);
    int vcm = mxGetM(mxvcol);
    int vcn = mxGetN(mxvcol);
    int opc = (int)mxGetScalar(mxopc);
    
    // check size
    if (vrm != 1 || vrn != n)
        mexErrMsgTxt("The size of the row vector is illegal, it should be 1 x n.\n");
    if (vcm != m || vcn != 1)
        mexErrMsgTxt("The size of the column vector is illegal, it should be m x 1.\n");        
    
    // get base address
    const double* X = mxGetPr(mxX);
    const double* vrow = mxGetPr(mxvrow);
    const double* vcol = mxGetPr(mxvcol);
    
    mxArray* mxY = mxCreateDoubleMatrix(m, n, mxREAL);
    double* Y = mxGetPr(mxY);
    
    // do operation
    delegate_operations(opc, X, vrow, vcol, Y, m, n);
    
    // make output
    plhs[0] = mxY;
    
}






