#include <iostream>
#include <random>
#include "distance.h"

int8_t* createVector(int size) {
  auto p = new int8_t[size];
  for (int i = 0; i < size; i++) {
    p[i] = 0;
  }
  return p;
}

void main(int argc, char** argv) {
  int8_t* a = createVector(64);
  int8_t* b = createVector(64);
}