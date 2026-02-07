#include "inc/Helper/ThreadContext.h"

void* DummyGet() { return nullptr; }
void  DummySet(void* context) {}

typedef void* (*GetOmpContextFuncType)();
typedef void (*SetOmpContextFuncType)(void *);

GetOmpContextFuncType s_getOmpContextFunc = &DummyGet;
SetOmpContextFuncType s_setOmpContextFunc = &DummySet;

void SetContextFuncs(void *getter, void *setter)
{
    s_getOmpContextFunc = (GetOmpContextFuncType)getter;
    s_setOmpContextFunc = (SetOmpContextFuncType)setter;
}

OmpParallelContext::OmpParallelContext()
{
    m_savedContext = (*s_getOmpContextFunc)();
}
OmpParallelContext::OmpParallelContext(const OmpParallelContext& other)
{
    m_savedContext = other.m_savedContext;
    (*s_setOmpContextFunc)(m_savedContext);
}
OmpParallelContext::~OmpParallelContext()
{
    (*s_setOmpContextFunc)(m_savedContext);
}

DefaultThreadContext::DefaultThreadContext()
{
    m_savedContext = (*s_getOmpContextFunc)();
    (*s_setOmpContextFunc)(nullptr);
}

DefaultThreadContext::~DefaultThreadContext()
{
    (*s_setOmpContextFunc)(m_savedContext);
}

void* DefaultThreadContext::SavedContext()
{
    return m_savedContext;
}

ThreadContext::ThreadContext(void* context)
{
    m_savedContext = (*s_getOmpContextFunc)();
    (*s_setOmpContextFunc)(context);
}

ThreadContext::~ThreadContext()
{
    (*s_setOmpContextFunc)(m_savedContext);
}

SavePartitionContext::SavePartitionContext()
{
    // Save the current allocator address, but not do allocator switch
    m_savedContext = (*s_getOmpContextFunc)();
}

SavePartitionContext::~SavePartitionContext()
{
    // No switch, so nothing to do here
}

void *SavePartitionContext::SavedContext()
{
    return m_savedContext;
}
