#pragma once

#include <windows.h>
#include <esent.h>
#include <iostream>

const size_t MaxItems = 2'000;
const size_t MaxItemsForPivot = 400;
const size_t PqChunksCount = 10;
const size_t QueriesCount = 2;

static const char *szDatabase = "esent_demo.mdb";
static const char *szDatabaseJfm = "esent_demo.jfm";
static const char *szItemsTable = "items";
static const char *szPivotTable = "pivots";
static const char *szIdIndex = "IdIndex";
static const char *szIdField = "id";
static const char *szVectorField = "vector";
static const char *szPqField = "pq";
static const char *szPivotField = "pivot";

#define NO_GRBIT 0
#define CP_ANSI 1252

#define CallJ(fn, label)                                                                                               \
    {                                                                                                                  \
        if ((err = (fn)) < 0)                                                                                          \
        {                                                                                                              \
            goto label;                                                                                                \
        }                                                                                                              \
    }

#define Call(fn) CallJ(fn, HandleError)

#define HANDLE_ERROR                                        \
        HandleError : if (err < 0)                          \
        {                                                   \
            JetRollback(sesidT, NO_GRBIT);                  \
            std::cerr << "JET ERROR: " << err << std::endl; \
            throw 0;                                        \
        }


extern JET_INSTANCE instance;
extern JET_SESID sesidT;
extern JET_DBID dbidDatabase;
extern size_t Dimensions;
extern size_t PivotSize;

void do_create_schema();
void do_load_items(const char *data_path);
void do_generate_pivot_data();
void do_compress_items();
void do_compress_queries(const char *query_path);

