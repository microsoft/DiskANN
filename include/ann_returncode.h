// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <string>
#include "windows_customizations.h"

namespace diskann
{

class ANNReturnCode
{
  public:
    enum class Value
    {
        SUCCESS,
        INVALID_LABEL
    };

    DISKANN_DLLEXPORT const char *getErrorDescription()
    {
        switch (_returncode)
        {
        case Value::SUCCESS:
            return "Success";
        case Value::INVALID_LABEL:
            return "Invalid Label provided for search";
        default:
            return "Unknown error";
        }
    }
    DISKANN_DLLEXPORT Value getReturnCode()
    {
        return _returncode;
    }

    ANNReturnCode() : _returncode(ANNReturnCode::Value::SUCCESS)
    {
    }
    ANNReturnCode(Value value) : _returncode(value)
    {
    }

  private:
    Value _returncode;
};

} // namespace diskann
