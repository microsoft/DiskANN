#pragma once

#ifndef __NSG_WINDOWS__
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#else
#include <Windows.h>
#endif
#include<string>

namespace NSG {
	class MemoryMapper {
	private:
	#ifndef __NSG_WINDOWS__
		int _fd;
	#else
		HANDLE _bareFile;
		HANDLE _fd;

	#endif
		char* _buf;
		size_t _fileSize;
		const char* _fileName;

	public:
		MemoryMapper(const char* filename);
		MemoryMapper(const std::string& filename);

		char* getBuf();
		size_t getFileSize();

		~MemoryMapper();
	};
}