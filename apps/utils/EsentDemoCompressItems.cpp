#include <vector>
#include "EsentDemo.h"
#include "pq.h"

static JET_ERR err = JET_errSuccess;
static JET_TABLEID items_table = JET_tableidNil;
static JET_COLUMNID vector_column_id = 0;
static JET_TABLEID pivots_table = JET_tableidNil;
static JET_COLUMNID pivot_column_id = 0;
static JET_COLUMNID pq_column_id = 0;
static size_t update_count = 0;

static void open_table();
static void load_pivot(std::vector<float> &pivot);
static void load_vectors(std::vector<float> &data, size_t from, size_t count);
static void store_pq(const std::vector<uint8_t>& pq, size_t from, size_t size);
static void close_table();

void do_compress_items()
{
    const size_t count = 100;

    open_table();

    std::vector<float> pivot;
    load_pivot(pivot);

    std::vector<float> data;
    std::vector<uint8_t> pq;
    for (size_t from = 0; ; from += count)
    {
        load_vectors(data, from, count);
        if (data.empty())
        {
            break;
        }

        size_t size = data.size() / Dimensions;

        pq.clear();
        diskann::generate_pq_data_from_pivots_mpopov(&data[0], size, 
            &pivot[0], pivot.size(), 
            PqChunksCount, Dimensions, 
            pq);

        store_pq(pq, from, size);
    }

    std::cout << "Stored " << update_count << " compressed vectors." << std::endl;

    close_table();
}

void open_table()
{
    JET_COLUMNDEF columndefT;

    Call(JetOpenTable(sesidT, dbidDatabase, szItemsTable, NULL, 0, 0L, &items_table));
    Call(JetGetTableColumnInfo(sesidT, items_table, szVectorField, &columndefT, sizeof(columndefT), JET_ColInfo));
    vector_column_id = columndefT.columnid;
    Call(JetGetTableColumnInfo(sesidT, items_table, szPqField, &columndefT, sizeof(columndefT), JET_ColInfo));
    pq_column_id = columndefT.columnid;

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

void load_pivot(std::vector<float> &pivot)
{
    pivot.resize(PivotSize);

    JET_RETRIEVECOLUMN rgretrievecolumnT[1];

    rgretrievecolumnT[0].columnid = pivot_column_id;
    rgretrievecolumnT[0].pvData = (void *)&pivot[0];
    rgretrievecolumnT[0].cbData = (unsigned long)(sizeof(float) * pivot.size());
    rgretrievecolumnT[0].cbActual = 0;
    rgretrievecolumnT[0].grbit = NO_GRBIT;
    rgretrievecolumnT[0].ibLongValue = 0;
    rgretrievecolumnT[0].itagSequence = 1;
    rgretrievecolumnT[0].columnidNextTagged = 0;
    rgretrievecolumnT[0].err = JET_errSuccess;

    Call(JetBeginTransaction(sesidT));

    for (err = JetMove(sesidT, pivots_table, JET_MoveFirst, NO_GRBIT);
         JET_errNoCurrentRecord != err;
         err = JetMove(sesidT, pivots_table, JET_MoveNext, NO_GRBIT))
    {
        Call(err);
        Call(JetRetrieveColumns(sesidT, pivots_table, rgretrievecolumnT, 1));
    }

    Call(JetCommitTransaction(sesidT, NO_GRBIT));

    std::cout << "Loaded pivot data with size " << PivotSize << std::endl;

    HANDLE_ERROR;
}

void load_vectors(std::vector<float>& data, size_t from, size_t count)
{
    JET_RETRIEVECOLUMN rgretrievecolumnT[1];
    std::vector<float> vec(Dimensions);
    size_t n = 0;
    long key = (long)from + 1;

    data.clear();
    data.reserve(Dimensions * count);

    rgretrievecolumnT[0].columnid = vector_column_id;
    rgretrievecolumnT[0].pvData = (void *)&vec[0];
    rgretrievecolumnT[0].cbData = (unsigned long)(sizeof(float) * vec.size());
    rgretrievecolumnT[0].cbActual = 0;
    rgretrievecolumnT[0].grbit = NO_GRBIT;
    rgretrievecolumnT[0].ibLongValue = 0;
    rgretrievecolumnT[0].itagSequence = 1;
    rgretrievecolumnT[0].columnidNextTagged = 0;
    rgretrievecolumnT[0].err = JET_errSuccess;

    Call(JetBeginTransaction(sesidT));

    Call(JetMakeKey(sesidT, items_table, &key, sizeof(key), JET_bitNewKey));

    for (err = JetSeek(sesidT, items_table, JET_bitSeekEQ);
         JET_errNoCurrentRecord != err && n < count;
         err = JetMove(sesidT, items_table, JET_MoveNext, NO_GRBIT), n++)
    {
        if (err == JET_errRecordNotFound)
        {
            break;
        }

        Call(err);
        Call(JetRetrieveColumns(sesidT, items_table, rgretrievecolumnT, 1));

        data.insert(data.end(), vec.begin(), vec.end());
    }

    Call(JetCommitTransaction(sesidT, NO_GRBIT));

    HANDLE_ERROR;
}

void store_pq(const std::vector<uint8_t>& pq, size_t from, size_t size)
{
    JET_SETCOLUMN rgsetcolumnT[1];
    size_t n = 0;
    long key = (long)from + 1;

    rgsetcolumnT[0].columnid = pq_column_id;
    rgsetcolumnT[0].cbData = PqChunksCount;
    rgsetcolumnT[0].grbit = NO_GRBIT;
    rgsetcolumnT[0].ibLongValue = 0;
    rgsetcolumnT[0].itagSequence = 1;
    rgsetcolumnT[0].err = JET_errSuccess;

    Call(JetBeginTransaction(sesidT));

    Call(JetMakeKey(sesidT, items_table, &key, sizeof(key), JET_bitNewKey));

    for (err = JetSeek(sesidT, items_table, JET_bitSeekEQ); JET_errNoCurrentRecord != err && n < size;
         err = JetMove(sesidT, items_table, JET_MoveNext, NO_GRBIT), n++)
    {
        rgsetcolumnT[0].pvData = (void *)(&pq[0] + n * PqChunksCount);

        Call(JetPrepareUpdate(sesidT, items_table, JET_prepReplace));
        Call(JetSetColumns(sesidT, items_table, rgsetcolumnT, 1));
        Call(JetUpdate(sesidT, items_table, NULL, 0, NULL));

        update_count++;
    }

    Call(JetCommitTransaction(sesidT, NO_GRBIT));

    HANDLE_ERROR;
}
