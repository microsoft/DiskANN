#pragma once
#include <windows.h>

// Allows creation of APIs that do no need output parameters
// (an alternative to exceptions)

// eg. HResult<dword> GetSomeNumber();
//      
//     HResult<dword> result = GetSomeString();
//     if (!Failed(result))
//     {
//        cout << "Succeded with result " << result.value;
//     }
//     else
//     {
//        cout << "Failed with hr: " << result.hr;
//     }


template<class T> struct HResult
{
    HRESULT hr;
    T value;
    
    HResult() 
        
        : hr(E_FAIL), value(T())
    {
    
    }
    
    HResult
    (
        T successValue
    ) 
    
        : hr(S_OK), value(successValue)
    {
    }
    
    HResult
    (
        HRESULT hr_,
        T value_
    ) 
        : hr(hr_), value(value_)
    {
    
    }
    
    static HResult Failure
    (
        HRESULT hr
    )
    {
        _ASSERT(FAILED(hr));

        HResult<T> hresult(hr, T());
        return hresult;
    }

    T GetValue
    (
        CONST T& defaultValue = T()
    ) CONST
    {
        return FAILED(hr) ? defaultValue : value;
    }
};

template<class T> static BOOL Failed
(
    CONST HResult<T> &result
)
{
    return FAILED(result.hr);
}




   
