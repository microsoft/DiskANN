#include <vector>
#include "EsentDemo.h"
#include "pq.h"

static JET_ERR err = JET_errSuccess;
static JET_TABLEID pivots_table = JET_tableidNil;
static JET_COLUMNID pivot_column_id = 0;

static void open_table();
static void load_pivot(std::vector<float> &pivot);
static void close_table();

void do_compress_queries(const char *query_path)
{
    open_table();

    std::vector<float> pivot;
    load_pivot(pivot);

    FILE *f;
    auto ret = fopen_s(&f, query_path, "rb");
    if (ret != 0)
    {
        std::cerr << "Failed to open data file " << query_path << std::endl;
        throw 0;
    }

    uint32_t header[2];
    size_t n = fread_s(header, sizeof(header), sizeof(*header), 2, f);
    if (n != 2)
    {
        std::cerr << "Failed to read header from data file " << query_path << std::endl;
    }

    uint32_t count = header[0];
    uint32_t dim = header[1];

    if (dim != Dimensions)
    {
        std::cerr << "Query dimensions do not match data dimensions.";
        throw 0;
    }

    std::cout << "Loading queries from file " << query_path << std::endl;
    std::cout << "\tDimensions: " << dim << std::endl;

    std::vector<uint8_t> pq;
    pq.reserve(PqChunksCount);

    std::vector<float> vec(dim);
    uint32_t i = 0;
    for (; i < count && i < QueriesCount; i++)
    {
        n = fread_s(&vec[0], dim * sizeof(float), sizeof(float), dim, f);
        if (n != dim)
        {
            std::cerr << "Failed to read vector #" << i + 1 << std::endl;
            throw 0;
        }

        pq.clear();
        diskann::generate_pq_data_from_pivots_mpopov(
            &vec[0], 1, 
            &pivot[0], pivot.size(), 
            PqChunksCount, Dimensions,
            pq);

        std::cout << "Quantized query #" << i + 1 << ": ";
        for (size_t j = 0; j < pq.size(); j++)
        {
            uint32_t t = pq[j];
            std::cout << t << "\t";
        }
        std::cout << std::endl;
    }

    fclose(f);

    close_table();
}

void open_table()
{
    JET_COLUMNDEF columndefT;

    Call(JetOpenTable(sesidT, dbidDatabase, szPivotTable, NULL, 0, 0L, &pivots_table));
    Call(JetGetTableColumnInfo(sesidT, pivots_table, szPivotField, &columndefT, sizeof(columndefT), JET_ColInfo));
    pivot_column_id = columndefT.columnid;

    HANDLE_ERROR;
}

void close_table()
{
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
