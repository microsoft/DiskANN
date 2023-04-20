// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <string>
#include "windows_customizations.h"

namespace diskann
{

class ANNErrorCode
{
  public:
    enum class Value
    {
        SUCCESS,
        INVALID_LABEL
    };

    DISKANN_DLLEXPORT const char *getErrorDescription()
    {
        switch (_errorCode)
        {
        case Value::SUCCESS:
            return "Success";
        case Value::INVALID_LABEL:
            return "Invalid Label provided for search";
        default:
            return "Unknown error";
        }
    }
    DISKANN_DLLEXPORT Value getErrorCode()
    {
        return _errorCode;
    }

    ANNErrorCode() : _errorCode(ANNErrorCode::Value::SUCCESS)
    {
    }
    ANNErrorCode(Value value) : _errorCode(value)
    {
    }

  private:
    Value _errorCode;
};

} // namespace diskann
