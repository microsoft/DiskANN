#include <jni.h>
#include "jni/basic_file_reader.h"

#include <iostream>

BasicFileReader::BasicFileReader(JNIEnv& env) : env(env) {
  this->javaClass = env.FindClass("FileInputStreamMediator");
  if (this->javaClass == nullptr) {
    std::cerr << "ERROR: Java class FileInputStreamMediator not found" << std::endl;
  }
  this->javaConstructor = this->env.GetMethodID(this->javaClass, "<init>", "(Ljava/lang/String;)V");
  // this would be a great place to verify the methods and arguments we expect to pass in are meeting our criteria
  if (this->javaConstructor == nullptr) {
    std::cerr << "ERROR: Java class FileInputStreamMediator does not comply with expected interface (ctor)" << std::endl;
  }
  this->javaClose = this->env.GetMethodID(this->javaClass, "close", "()V");
  if (this->javaClose == nullptr) {
    std::cerr << "ERROR: Java class FileInputStreamMediator does not comply with expected interface (close)" << std::endl;
  }
  // https://docs.oracle.com/javase/1.5.0/docs/guide/jni/spec/types.html#wp276 you'll need this. trust me.
  this->javaRead = this->env.GetMethodID(this->javaClass, "read", "([I[I[J)Z");
  if (this->javaRead == nullptr) {
    std::cerr << "ERROR: Java class FileInputStreamMediator does not comply with expected interface (read)" << std::endl;
  }
}

BasicFileReader::~BasicFileReader() {
  // I do nothing
}

void BasicFileReader::open(const std::string &fname) {
  jobject fileName = this->env.NewStringUTF(fname.c_str());
  this->jniFileReader = this->env.NewObject(this->javaClass, this->javaConstructor, fileName);
}

void BasicFileReader::close() {
  jboolean foo = this->env.CallBooleanMethod(this->jniFileReader, this->javaClose);
  if (this->env.ExceptionOccurred()) {
    std::cerr << "Things went wrong when we closed our JNI File Reader: " << std::endl;
    this->env.ExceptionDescribe();
    this->env.ExceptionClear();
    exit(-1);
  }
}

void BasicFileReader::read(std::vector<AlignedRead> &read_reqs) {
  jsize size = (jsize)read_reqs.size();
  jintArray readFroms = this->env.NewIntArray(size);
  jint readFromValues[size];

  jintArray readUntils = this->env.NewIntArray(size);
  jint readUntilValues[size];

  jlongArray readIntos = this->env.NewLongArray(size);
  jlong readIntoValues[size];
  for (size_t i = 0; i < read_reqs.size(); i++) {
    readFromValues[i] = (jint)read_reqs[i].offset;
    readUntilValues[i] = (jint)read_reqs[i].len; // readUntil could be named better
    readIntoValues[i] = (jlong)read_reqs[i].buf;
  }

  this->env.SetIntArrayRegion(readFroms, 0, size, readFromValues);
  this->env.SetIntArrayRegion(readUntils, 0, size, readUntilValues);
  this->env.SetLongArrayRegion(readIntos, 0, size, readIntoValues);

  // so now we can finally call the dang method
  this->env.CallVoidMethod(this->jniFileReader, this->javaRead, readFroms, readUntils, readIntos);
}