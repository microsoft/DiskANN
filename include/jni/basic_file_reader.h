#include <jni.h>
#include "aligned_file_reader.h"

class BasicFileReader { // should properly implement the AlignedFileReader, but I genuinely don't know what it's trying to achieve with all the threading in it
 private:
  JNIEnv &env;
  jclass javaClass;
  jmethodID javaConstructor;
  jmethodID javaRead;
  jmethodID javaClose;
  jobject jniFileReader;
 public:
  BasicFileReader(JNIEnv &env);
  ~BasicFileReader();

  void open(const std::string &fname);
  void close();

  void read(std::vector<AlignedRead> &read_reqs);

};
