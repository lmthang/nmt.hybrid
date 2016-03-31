// The mex function to generate GUID (Global Unique Identifier)
//
// This function is based on Win32 Platform SDK on COM. 
// It is essentaily wrapping the API function: CoCreateGuid
//
// History:
//  - Created by Dahua Lin, on Aug 12nd, 2006
//

#include <memory.h>
#include <ObjBase.h>

#include "mex.h"


//
// No Input
// Output two variables
//  - output var 1:  logical, indicate whether succeed
//  - output var 2:  a 1 x 16 uint8 array storing the 128-bit GUID
//                   if not successful, it is an empty array
//
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{            
    if (nrhs > 0)
    {
        mexErrMsgTxt("No input variables are allowed for win32guid_core");
    }
    if (nlhs != 2)
    {
        mexErrMsgTxt("The number of outputs should be exactly 2");
    }
    
    // generate the GUID
    GUID gid;
    HRESULT ret = CoCreateGuid(&gid);
        
    // output
	if (ret == S_OK)
    {
        mxArray *parrGuid = mxCreateNumericMatrix(1, 16, mxUINT8_CLASS, mxREAL);
        unsigned char* pBuffer = (unsigned char*)mxGetData(parrGuid);
        
        memcpy(pBuffer, &gid, 16);
        
        mxArray *pRet = mxCreateLogicalScalar((mxLogical)1);
        
        plhs[0] = pRet;
        plhs[1] = parrGuid;
    }
    else
    {
        mxArray *parrGuid = mxCreateNumericMatrix(0, 0, mxUINT8_CLASS, mxREAL);
        mxArray *pRet = mxCreateLogicalScalar((mxLogical)0);
        
        plhs[0] = pRet;
        plhs[1] = parrGuid;
    }
}

