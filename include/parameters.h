// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <sstream>
#include <typeinfo>
#include <unordered_map>

namespace diskann {

  class Parameters {
   public:
    Parameters() {
      int *p = new int;
      *p = 0;
      params["num_threads"] = p;
    }

    template<typename ParamType>
    inline void Set(const std::string &name, const ParamType &value) {
      //      ParamType *ptr = (ParamType *) malloc(sizeof(ParamType));
      if (params.find(name) != params.end()) {
        free(params[name]);
      }
      ParamType *ptr = new ParamType;
      *ptr = value;
      params[name] = (void *) ptr;
    }

    template<typename ParamType>
    inline ParamType Get(const std::string &name) const {
      auto item = params.find(name);
      if (item == params.end()) {
        throw std::invalid_argument("Invalid parameter name.");
      } else {
        // return ConvertStrToValue<ParamType>(item->second);
        if (item->second == nullptr) {
          throw std::invalid_argument(std::string("Parameter ") + name +
                                      " has value null.");
        } else {
          return *(static_cast<ParamType *>(item->second));
        }
      }
    }

    template<typename ParamType>
    inline ParamType Get(const std::string &name,
                         const ParamType &  default_value) {
      try {
        return Get<ParamType>(name);
      } catch (std::invalid_argument& e) {
        return default_value;
      }
    }

    ~Parameters() {
      for (auto iter = params.begin(); iter != params.end(); iter++) {
        if (iter->second != nullptr)
          free(iter->second);
        // delete iter->second;
      }
    }

   private:
    std::unordered_map<std::string, void *> params;

    Parameters(const Parameters &);
    Parameters &operator=(const Parameters &);

    template<typename ParamType>
    inline ParamType ConvertStrToValue(const std::string &str) const {
      std::stringstream sstream(str);
      ParamType         value;
      if (!(sstream >> value) || !sstream.eof()) {
        std::stringstream err;
        err << "Failed to convert value '" << str
            << "' to type: " << typeid(value).name();
        throw std::runtime_error(err.str());
      }
      return value;
    }
  };
}  // namespace diskann
