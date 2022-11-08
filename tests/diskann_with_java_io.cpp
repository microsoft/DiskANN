//
// Created by dax on 11/7/22.
//

#include <jni.h>
#include <iostream>
#include "jni/basic_file_reader.h"
#include "aligned_file_reader.h"

int main()
{
  JavaVM *jvm;
  JNIEnv *env;
  JavaVMInitArgs vm_args;
  JavaVMOption* options = new JavaVMOption[1];
  options[0].optionString = "-Djava.class.path=/home/dax/dev/DiskANN/java/";
  vm_args.version = JNI_VERSION_10; // we probably need a much newer version than this
  vm_args.nOptions = 1;
  vm_args.options = options;
  vm_args.ignoreUnrecognized = false;
  jint rc = JNI_CreateJavaVM(&jvm, (void**)&env, &vm_args);
  delete options;
  if (rc != JNI_OK) {
    std::cout << "Something went wrong creating the JavaVM for DiskANN: " << rc << std::endl;
  }
  // here's where the work goes

  std::vector<AlignedRead> reads;
  AlignedRead aligned;
  aligned.offset = 2*512;
  aligned.len = 10*512;
  aligned.buf = (void *) (10000*512); // a fake buffer that we never allocated and I'm really glad our java reader
                                      // doesn't actually try to write here yet
  reads.push_back(aligned);
  BasicFileReader reader(*env);
  reader.open("/home/dax/dev/DiskANN/CMakeLists.txt");
  reader.read(reads);

  // and here's where we clean up
  jvm->DestroyJavaVM();
}
