
#include <iostream>
#include <vector>
#include "EsentDemo.h"

static JET_ERR err = JET_errSuccess;
static JET_TABLEID items_table = JET_tableidNil;
static JET_COLUMNID vector_column_id = 0;
static bool in_transaction = false;

static void open_table();
static void insert_row(uint32_t n, std::vector<float> &vec);
static void close_table();

void do_load_items(const char *data_path)
{
    open_table();

    FILE *f = nullptr;
    auto ret = fopen_s(&f, data_path, "rb");
    if (ret != 0)
    {
        std::cerr << "Failed to open data file " << data_path << std::endl;
        throw 0;
    }

    uint32_t header[2];
    size_t n = fread_s(header, sizeof(header), sizeof(*header), 2, f);
    if (n != 2)
    {
        std::cerr << "Failed to read header from data file " << data_path << std::endl;
    }

    uint32_t count = header[0];
    uint32_t dim = header[1];
    Dimensions = dim;

    std::cout << "Loading data from file " << data_path << std::endl;
    std::cout << "\tDimensions: " << dim << std::endl;
    std::cout << "\tCount: " << count << std::endl;

    std::vector<float> vec(dim);
    uint32_t i = 0;
    for (; i < count && i < MaxItems; i++)
    {
        n = fread_s(&vec[0], dim * sizeof(float), sizeof(float), dim, f);
        if (n != dim)
        {
            std::cerr << "Failed to read vector #" << i+1 << std::endl;
            throw 0;
        }

        insert_row(i, vec);
    }

    fclose(f);
    close_table();

    std::cout << "Loaded " << i << " items." << std::endl;
}

void open_table()
{
    JET_COLUMNDEF columndefT;
    
    Call(JetOpenTable(sesidT, dbidDatabase, szItemsTable, NULL, 0, 0L, &items_table));

    Call(JetGetTableColumnInfo(sesidT, items_table, szVectorField, &columndefT, sizeof(columndefT), JET_ColInfo));
    vector_column_id = columndefT.columnid;

    HANDLE_ERROR;
}

void close_table()
{
    if (in_transaction)
    {
        Call(JetCommitTransaction(sesidT, NO_GRBIT));
        in_transaction = false;
    }

    JetCloseTable(sesidT, items_table);
    HANDLE_ERROR;
}

void insert_row(uint32_t n, std::vector<float>& vec)
{
    JET_SETCOLUMN rgsetcolumnT[1];

    if (!in_transaction)
    {
        Call(JetBeginTransaction(sesidT));
        in_transaction = true;
    }

    Call(JetPrepareUpdate(sesidT, items_table, JET_prepInsert));

    rgsetcolumnT[0].columnid = vector_column_id;
    rgsetcolumnT[0].pvData = (void *)&vec[0];
    rgsetcolumnT[0].cbData = (unsigned long)(vec.size() * sizeof(float));
    rgsetcolumnT[0].grbit = NO_GRBIT;
    rgsetcolumnT[0].ibLongValue = 0;
    rgsetcolumnT[0].itagSequence = 1;
    rgsetcolumnT[0].err = JET_errSuccess;

    Call(JetSetColumns(sesidT, items_table, rgsetcolumnT, 1));
    Call(JetUpdate(sesidT, items_table, NULL, 0, NULL));

    if (n > 0 && n % 1000)
    {
        Call(JetCommitTransaction(sesidT, NO_GRBIT));
        in_transaction = false;
    }

    HANDLE_ERROR;
}
