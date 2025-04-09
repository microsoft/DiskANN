#pragma once

#ifndef _THREAD_CONTEXT_H_
#define _THREAD_CONTEXT_H_

void SetContextFuncs(void *getter, void *setter);

class OmpParallelContext
{
public:
    OmpParallelContext();
    OmpParallelContext(const OmpParallelContext& other);
    ~OmpParallelContext();

private:
    void* m_savedContext;
};

class DefaultThreadContext
{
public:
    DefaultThreadContext();
    ~DefaultThreadContext();

    void* SavedContext();

private:
    void* m_savedContext;
};

class ThreadContext
{
public:
    ThreadContext(void* context);
    ~ThreadContext();

private:
    void* m_savedContext;
};

class SavePartitionContext
{
public:
    SavePartitionContext();
    ~SavePartitionContext();
    void* SavedContext();

private:
    void* m_savedContext;
};
#endif