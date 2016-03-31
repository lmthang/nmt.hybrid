///////////////////////////////////////////////////////////
//
//  The implementation of some core computational 
//  routines for pairwise histogram metric
//
//  History
//    - Created by Dahua Lin on Sep 19, 2006
//
////////////////////////////////////////////////////////////

#include "mex.h"
#include <math.h>

// Calculate pairwise histogram intersect
// H1: d x n1
// H2: d x n2
// D:  n1 x n2
void calc_intersect_pw(int n1, int n2, int d, const double* H1, const double* H2, double* D)
{
    // compute all sum h2(k) and cache
    const double* p2 = H2;
    double* sh2 = new double[n2];
    double s = 0;
    
    for (int j = 0; j < n2; ++j)
    {
        s = 0;
        for (int k = 0; k < d; ++k)
        {
            s += *(p2++);
        }
        sh2[j] = s;
    }
    
    // compute main part
    const double* h2 = H2;                   
    for (int j = 0; j < n2; ++j)
    {                
        const double* h1 = H1;
        for (int i = 0; i < n1; ++i)
        {
            // D(i,j) = 1- sum min(h1(k), h2(k)) / sum h2(k)            
            s = 0;
            for (int k = 0; k < d; ++k)
            {
                s += h1[k] < h2[k] ? h1[k] : h2[k];                                
            }
            *(D++) = 1 - s / sh2[j];
            
            h1 += d;
        }
        h2 += d;
    }
    
    delete[] sh2;
}

// Calculate Pairwise Chi-square distances
void calc_chisq_pw(int n1, int n2, int d, const double* H1, const double* H2, double* D)
{
    double s = 0;
    double vs = 0;
    double vd = 0;
   
    const double* h2 = H2;    
    for(int j = 0; j < n2; ++j)
    {
        const double* h1 = H1;
        for (int i = 0; i < n1; ++i)
        {
            // D(i, j) = sum (h1(k) - h2(k))^2 / (2 * (h1(k)+h2(k))
            s = 0;
            for (int k = 0; k < d; ++k)
            {
                vd = h1[k] - h2[k];
                vs = h1[k] + h2[k];
                s += vd * vd / (2 * vs);
            }
            *(D++) = s;
            
            h1 += d;
        }
        h2 += d;
    }
}

inline double mullog(double v)
{
    return v > 0 ? v * log(v) : 0; 
}


// Calculate Pairwise Jeffrey distances
void calc_jeffrey_pw(int n1, int n2, int d, const double* H1, const double* H2, double* D)
{
    double s = 0;
    double mv = 0;
   
    const double* h2 = H2;    
    for(int j = 0; j < n2; ++j)
    {
        const double* h1 = H1;
        for (int i = 0; i < n1; ++i)
        {
            // D(i, j) = sum (h1(k) - h2(k))^2 / (2 * (h1(k)+h2(k))
            s = 0;
            for (int k = 0; k < d; ++k)
            {
                mv = (h1[k] + h2[k]) / 2;
                s += (mullog(h1[k]) + mullog(h2[k]) - 2 * mullog(mv));
            }
            *(D++) = s;
            
            h1 += d;
        }
        h2 += d;
    }
}



// Main entry function
// Input
//  - H1, H2, mcode(1 - intersect, 2 -chisquare, 3-jeffrey)
// Output
//  - D
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // check input
    if (nrhs != 3)
    {
        mexErrMsgTxt("histmetricpw_core uses 3 input arguments.\n");
    }
    if (nlhs > 1)
    {
        mexErrMsgTxt("Too many output arguments for histmetricpw_core.\n");
    }
    
    // get input
    const mxArray* mxH1 = prhs[0];
    const mxArray* mxH2 = prhs[1];
    const mxArray* mxmcode = prhs[2];
    
    // verify input
    int d1 = mxGetM(mxH1);
    int n1 = mxGetN(mxH1);
    int d2 = mxGetM(mxH2);
    int n2 = mxGetN(mxH2);
    int mcode = (int)mxGetScalar(mxmcode);
    
    if (d1 != d2)
    {
        mexErrMsgTxt("The dimensions of H1 ans H2 are not equal.\n");
    }
    if (mcode < 1 || mcode > 3)
    {
        mexErrMsgTxt("mcode should be either 1 or 2.\n");
    }
    int d = d1;
    
    // prepare output
    mxArray* mxD = mxCreateDoubleMatrix(n1, n2, mxREAL);
    
    // get base addresses
    const double* H1 = mxGetPr(mxH1);
    const double* H2 = mxGetPr(mxH2);
    double* D = mxGetPr(mxD);
    
    // do computation
    if (mcode == 1)
    {
        calc_intersect_pw(n1, n2, d, H1, H2, D);
    }
    else if (mcode == 2)
    {
        calc_chisq_pw(n1, n2, d, H1, H2, D);
    }
    else
    {
        calc_jeffrey_pw(n1, n2, d, H1, H2, D);
    }
    
    // assign output
    plhs[0] = mxD;
    
}







