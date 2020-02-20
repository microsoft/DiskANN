#pragma once

template<typename T>
class aligned_dtor {
  void operator()(T* p) {
    aligned_free(p);
  }
};