/*******************************************************************************
* Copyright 2006-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * This header file describes how memory allocation can be replaced
 * in the Intel(R) Math Kernel Library (Intel(R) MKL). It contains
 * all declarations required by an application developer to replace
 * the memory allocation.
 *
 * Intel(R) MKL supporting this feature only use the following
 * functions to allocate or free memory:
 * - malloc
 * - calloc
 * - realloc
 * - free
 * and call those functions via globally visible function pointers:
 * - i_malloc
 * - i_calloc
 * - i_realloc
 * - i_free
 *
 * C++ new and delete operators are changed to also use these function
 * pointers, if they occur in a library at all. No library supporting memory 
 * allocation replacement will allocate memory before it is invoked explicitly
 * by the application for the first time.
 *
 * Therefore an application can safely set these function pointers
 * at the very beginning of its execution to some other replacement
 * functions. The function pointers must remain valid while
 * Intel(R) MKL is in use.
 *
 * Setting these pointers is optional because the copies contained in
 * Intel(R) MKL point to the standard C library functions by default.
 *
 * On Windows(R) data exported by a DLL and data contained in a static
 * library are accessed differently. To support mixing static
 * libraries and DLLs, the function pointers exist in two sets with
 * different names so that the application can override both sets in
 * the same source file without running into name conflicts.
 *
 * Here is an example:
\verbatim

   #include <i_malloc.h>
   #include <my_malloc.h>

   int main( int argc, int argv )
   {
       // override normal pointers
       i_malloc = my_malloc;
       i_calloc = my_calloc;
       i_realloc = my_realloc;
       i_free = my_free;

   #ifdef _WIN32
       // also override pointers used by DLLs
       i_malloc_dll = my_malloc;
       i_calloc_dll = my_calloc;
       i_realloc_dll = my_realloc;
       i_free_dll = my_free;
   #endif

   }
\endverbatim
 */

#ifndef _I_MALLOC_H_
#define _I_MALLOC_H_

#include <stdlib.h>     /* for size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* typedefs for all four function pointers */
typedef void * (* i_malloc_t)(size_t size);
typedef void * (* i_calloc_t)(size_t nmemb, size_t size);
typedef void * (* i_realloc_t)(void *ptr, size_t size);
typedef void (* i_free_t)(void *ptr);

#ifdef _WIN32

# ifdef INTEL_DLL_EXPORTS
#  define INTEL_API_DEF __declspec(dllexport)
# else
#  define INTEL_API_DEF __declspec(dllimport)
#endif

/* function pointers as used and exported by a DLL */
extern INTEL_API_DEF i_malloc_t i_malloc_dll;
extern INTEL_API_DEF i_calloc_t i_calloc_dll;
extern INTEL_API_DEF i_realloc_t i_realloc_dll;
extern INTEL_API_DEF i_free_t i_free_dll;

#else /* _WIN32 */

# define INTEL_API_DEF

#endif /* _WIN32 */

/* normal function pointers for static libraries */
extern i_malloc_t i_malloc;
extern i_calloc_t i_calloc;
extern i_realloc_t i_realloc;
extern i_free_t i_free;

#ifdef __cplusplus
}
#endif

#endif /* _I_MALLOC_H_ */
