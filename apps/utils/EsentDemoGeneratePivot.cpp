#include <iostream>
#include <vector>
#include "EsentDemo.h"
#include "pq.h"

static JET_ERR err = JET_errSuccess;
static JET_TABLEID items_table = JET_tableidNil;
static JET_COLUMNID vector_column_id = 0;
static JET_TABLEID pivots_table = JET_tableidNil;
static JET_COLUMNID pivot_column_id = 0;

static void open_table();
static void load_data(std::vector<float>& data, size_t& count);
static void store_pivot(std::vector<float> &pivot);
static void close_table();

void do_generate_pivot_data()
{
    open_table();

    size_t count = 0;
    std::vector<float> data;
    load_data(data, count);

    std::vector<float> pivot;
    diskann::generate_pq_pivots_mpopov(&data[0], count, Dimensions, PqChunksCount, pivot);
    PivotSize = pivot.size();

    store_pivot(pivot);

    close_table();
}

void open_table()
{
    JET_COLUMNDEF columndefT;

    Call(JetOpenTable(sesidT, dbidDatabase, szItemsTable, NULL, 0, 0L, &items_table));
    Call(JetGetTableColumnInfo(sesidT, items_table, szVectorField, &columndefT, sizeof(columndefT), JET_ColInfo));
    vector_column_id = columndefT.columnid;

    Call(JetOpenTable(sesidT, dbidDatabase, szPivotTable, NULL, 0, 0L, &pivots_table));
    Call(JetGetTableColumnInfo(sesidT, pivots_table, szPivotField, &columndefT, sizeof(columndefT), JET_ColInfo));
    pivot_column_id = columndefT.columnid;

    HANDLE_ERROR;
}

void close_table()
{
    JetCloseTable(sesidT, items_table);
    JetCloseTable(sesidT, pivots_table);
}

void load_data(std::vector<float>& data, size_t& count)
{
    JET_RETRIEVECOLUMN rgretrievecolumnT[1];
    std::vector<float> vec(Dimensions);

    rgretrievecolumnT[0].columnid = vector_column_id;
    rgretrievecolumnT[0].pvData = (void*)&vec[0];
    rgretrievecolumnT[0].cbData = (unsigned long)(sizeof(float) * vec.size());
    rgretrievecolumnT[0].cbActual = 0;
    rgretrievecolumnT[0].grbit = NO_GRBIT;
    rgretrievecolumnT[0].ibLongValue = 0;
    rgretrievecolumnT[0].itagSequence = 1;
    rgretrievecolumnT[0].columnidNextTagged = 0;
    rgretrievecolumnT[0].err = JET_errSuccess;

    Call(JetBeginTransaction(sesidT));

    for (err = JetMove(sesidT, items_table, JET_MoveFirst, NO_GRBIT); 
         JET_errNoCurrentRecord != err && count < MaxItemsForPivot;
         err = JetMove(sesidT, items_table, JET_MoveNext, NO_GRBIT), count++)
    {
        Call(err);
        Call(JetRetrieveColumns(sesidT, items_table, rgretrievecolumnT, 1));

        data.insert(data.end(), vec.begin(), vec.end());
    }

    Call(JetCommitTransaction(sesidT, NO_GRBIT));

    std::cout << "Loaded " << count << " items for generating pivot." << std::endl;

    HANDLE_ERROR;
}

void store_pivot(std::vector<float>& pivot)
{
    JET_SETCOLUMN rgsetcolumnT[1];

    Call(JetBeginTransaction(sesidT));
    Call(JetPrepareUpdate(sesidT, pivots_table, JET_prepInsert));

    rgsetcolumnT[0].columnid = pivot_column_id;
    rgsetcolumnT[0].pvData = (void *)&pivot[0];
    rgsetcolumnT[0].cbData = (unsigned long)(pivot.size() * sizeof(float));
    rgsetcolumnT[0].grbit = NO_GRBIT;
    rgsetcolumnT[0].ibLongValue = 0;
    rgsetcolumnT[0].itagSequence = 1;
    rgsetcolumnT[0].err = JET_errSuccess;

    Call(JetSetColumns(sesidT, pivots_table, rgsetcolumnT, 1));
    Call(JetUpdate(sesidT, pivots_table, NULL, 0, NULL));
    Call(JetCommitTransaction(sesidT, NO_GRBIT));

    HANDLE_ERROR;
}

