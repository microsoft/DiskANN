
#include <iostream>
#include "EsentDemo.h"

JET_INSTANCE instance = 0;
JET_SESID sesidT = JET_sesidNil;
JET_DBID dbidDatabase = JET_dbidNil;
size_t Dimensions = 0;
size_t PivotSize = 0;

void start_jet();
void create_schema();
void load_items(const char* data_path);
void generate_pivot_data();
void compress_items();
void compress_queries(const char* query_path);
void stop_jet();

BOOL FileExists(const char *szPath);

void esent_demo(const char *data_path, const char *query_path)
{
    start_jet();
    create_schema();
    load_items(data_path);
    generate_pivot_data();
    compress_items();
    compress_queries(query_path);
    stop_jet();
}

void start_jet()
{
    JET_ERR err = JET_errSuccess;
    bool is_init = false;

    Call(JetSetSystemParameter(&instance, 0, JET_paramSystemPath, 0, ".\\"));
    Call(JetSetSystemParameter(&instance, 0, JET_paramTempPath, 0, ".\\"));
    Call(JetSetSystemParameter(&instance, 0, JET_paramLogFilePath, 0, ".\\"));
    Call(JetSetSystemParameter(&instance, 0, JET_paramBaseName, 0, "edb"));
    Call(JetSetSystemParameter(&instance, 0, JET_paramCircularLog, 1, NULL));

    is_init = true;
    Call(JetInit(&instance));
    is_init = false;

    Call(JetBeginSession(instance, &sesidT, nullptr, nullptr));

HandleError:
    if (err < 0)
    {
        if (is_init)
        {
            std::cerr << "Failed to init JET. You must delete all edb* files." << std::endl;
        }
        JetRollback(sesidT, NO_GRBIT);
        throw 0;
    }
}

void stop_jet()
{
    JetCloseDatabase(sesidT, dbidDatabase, NO_GRBIT);
    JetEndSession(sesidT, NO_GRBIT);
    JetTerm(instance);
}

void create_schema()
{
    std::cout << "Creating schema..." << std::endl;

    if (FileExists(szDatabase))
    {
        if (!DeleteFileA(szDatabase))
        {
            std::cerr << "Failed to delete file " << szDatabase << std::endl;
            throw 0;
        }

        if (!DeleteFileA(szDatabaseJfm))
        {
            std::cerr << "Failed to delete file " << szDatabaseJfm << std::endl;
            throw 0;
        }
    }

    do_create_schema();

    std::cout << "Creating schema done" << std::endl << std::endl;
}

void load_items(const char* data_path)
{
    std::cout << "Loading items form " << data_path << "... " << std::endl;
    do_load_items(data_path);
    std::cout << "Loading items done" << std::endl << std::endl;
}

void generate_pivot_data()
{
    std::cout << "Generating pivot data..." << std::endl;
    do_generate_pivot_data();
    std::cout << "Generating pivot data done" << std::endl << std::endl;
}

void compress_items()
{
    std::cout << "Generating pivot data..." << std::endl;
    do_compress_items();
    std::cout << "Generating pivot data done" << std::endl << std::endl;
}

void compress_queries(const char* query_path)
{
    std::cout << "Generating PQ for queries " << query_path << "... " << std::endl;
    do_compress_queries(query_path);
    std::cout << "Generating PQ for queries done" << std::endl << std::endl;
}

BOOL FileExists(const char *szPath)
{
    DWORD dwAttrib = GetFileAttributesA(szPath);
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}
