
#include <iostream>
#include "EsentDemo.h"

void create_items_table(JET_SESID sesidT, JET_DBID dbidDatabase);
void create_pivot_table(JET_SESID sesidT, JET_DBID dbidDatabase);

void do_create_schema()
{
    JET_ERR err = JET_errSuccess;

    err = JetAttachDatabase(sesidT, szDatabase, 0);
    if (err == JET_errSuccess)
    {
        std::cerr << "Database " << szDatabase << " already exists" << std::endl;
        throw 0;
    }
    
    if (JET_errFileNotFound != err)
    {   
        std::cerr << "Database " << szDatabase << " is in bad state" << std::endl;
        throw 0;
    }
    
    Call(JetCreateDatabase(sesidT, szDatabase, nullptr, &dbidDatabase, 0));

    Call(JetBeginTransaction(sesidT));
    create_items_table(sesidT, dbidDatabase);
    create_pivot_table(sesidT, dbidDatabase);
    Call(JetCommitTransaction(sesidT, NO_GRBIT));

HandleError:
    if (err < 0)
    {
        JetRollback(sesidT, NO_GRBIT);
        throw 0;
    }
}

void create_items_table(JET_SESID sesidT, JET_DBID dbidDatabase)
{
    JET_ERR err = JET_errSuccess;
    JET_TABLECREATE tablecreateT;
    JET_COLUMNCREATE rgcolumncreateT[3];
    JET_INDEXCREATE rgindexcreateT[1];

    rgcolumncreateT[0].cbStruct = sizeof(rgcolumncreateT[0]);
    rgcolumncreateT[0].szColumnName = (char*)szIdField;
    rgcolumncreateT[0].coltyp = JET_coltypLong;
    rgcolumncreateT[0].cbMax = 0;
    rgcolumncreateT[0].grbit = JET_bitColumnAutoincrement;
    rgcolumncreateT[0].pvDefault = NULL;
    rgcolumncreateT[0].cbDefault = 0;
    rgcolumncreateT[0].cp = CP_ANSI;
    rgcolumncreateT[0].columnid = 0;
    rgcolumncreateT[0].err = JET_errSuccess;

    rgcolumncreateT[1].cbStruct = sizeof(rgcolumncreateT[1]);
    rgcolumncreateT[1].szColumnName = (char*)szVectorField;
    rgcolumncreateT[1].coltyp = JET_coltypLongBinary;
    rgcolumncreateT[1].cbMax = 0;
    rgcolumncreateT[1].grbit = NO_GRBIT;
    rgcolumncreateT[1].pvDefault = NULL;
    rgcolumncreateT[1].cbDefault = 0;
    rgcolumncreateT[1].cp = CP_ANSI;
    rgcolumncreateT[1].columnid = 0;
    rgcolumncreateT[1].err = JET_errSuccess;

    rgcolumncreateT[2].cbStruct = sizeof(rgcolumncreateT[2]);
    rgcolumncreateT[2].szColumnName = (char *)szPqField;
    rgcolumncreateT[2].coltyp = JET_coltypBinary;
    rgcolumncreateT[2].cbMax = 0;
    rgcolumncreateT[2].grbit = NO_GRBIT;
    rgcolumncreateT[2].pvDefault = NULL;
    rgcolumncreateT[2].cbDefault = 0;
    rgcolumncreateT[2].cp = CP_ANSI;
    rgcolumncreateT[2].columnid = 0;
    rgcolumncreateT[2].err = JET_errSuccess;

    char rgbIDIndex[] = "+id\0\0";
    int cchIDIndex = 5;

    rgindexcreateT[0].cbStruct = sizeof(rgindexcreateT[0]);
    rgindexcreateT[0].szIndexName = (char *)szIdIndex;
    rgindexcreateT[0].szKey = rgbIDIndex;
    rgindexcreateT[0].cbKey = cchIDIndex;
    rgindexcreateT[0].grbit = JET_bitIndexPrimary;
    rgindexcreateT[0].ulDensity = 100;
    rgindexcreateT[0].lcid = MAKELCID(0x409, SORT_DEFAULT);
    rgindexcreateT[0].cbVarSegMac = 0;
    rgindexcreateT[0].rgconditionalcolumn = NULL;
    rgindexcreateT[0].cConditionalColumn = 0;
    rgindexcreateT[0].err = JET_errSuccess;

    tablecreateT.cbStruct = sizeof(tablecreateT);
    tablecreateT.szTableName = (char *)szItemsTable;
    tablecreateT.szTemplateTableName = NULL;
    tablecreateT.ulPages = 16;
    tablecreateT.ulDensity = 80;
    tablecreateT.rgcolumncreate = rgcolumncreateT;
    tablecreateT.cColumns = 3;
    tablecreateT.rgindexcreate = rgindexcreateT;
    tablecreateT.cIndexes = 1;
    tablecreateT.grbit = NO_GRBIT;
    tablecreateT.tableid = JET_tableidNil;
    tablecreateT.cCreated = 0;

    Call(JetCreateTableColumnIndex(sesidT, dbidDatabase, &tablecreateT));
    Call(JetCloseTable(sesidT, tablecreateT.tableid));

HandleError:
    if (err < 0)
    {
        JetRollback(sesidT, 0);
        throw 0;
    }
}

void create_pivot_table(JET_SESID sesidT, JET_DBID dbidDatabase)
{
    JET_ERR err = JET_errSuccess;
    JET_TABLECREATE tablecreateT;
    JET_COLUMNCREATE rgcolumncreateT[2];
    JET_INDEXCREATE rgindexcreateT[1];

    rgcolumncreateT[0].cbStruct = sizeof(rgcolumncreateT[0]);
    rgcolumncreateT[0].szColumnName = (char *)szIdField;
    rgcolumncreateT[0].coltyp = JET_coltypLong;
    rgcolumncreateT[0].cbMax = 0;
    rgcolumncreateT[0].grbit = JET_bitColumnAutoincrement;
    rgcolumncreateT[0].pvDefault = NULL;
    rgcolumncreateT[0].cbDefault = 0;
    rgcolumncreateT[0].cp = CP_ANSI;
    rgcolumncreateT[0].columnid = 0;
    rgcolumncreateT[0].err = JET_errSuccess;

    rgcolumncreateT[1].cbStruct = sizeof(rgcolumncreateT[1]);
    rgcolumncreateT[1].szColumnName = (char *)szPivotField;
    rgcolumncreateT[1].coltyp = JET_coltypLongBinary;
    rgcolumncreateT[1].cbMax = 0;
    rgcolumncreateT[1].grbit = NO_GRBIT;
    rgcolumncreateT[1].pvDefault = NULL;
    rgcolumncreateT[1].cbDefault = 0;
    rgcolumncreateT[1].cp = CP_ANSI;
    rgcolumncreateT[1].columnid = 0;
    rgcolumncreateT[1].err = JET_errSuccess;

    char rgbIDIndex[] = "+id\0\0";
    int cchIDIndex = 5;

    rgindexcreateT[0].cbStruct = sizeof(rgindexcreateT[0]);
    rgindexcreateT[0].szIndexName = (char *)szIdIndex;
    rgindexcreateT[0].szKey = rgbIDIndex;
    rgindexcreateT[0].cbKey = cchIDIndex;
    rgindexcreateT[0].grbit = JET_bitIndexPrimary;
    rgindexcreateT[0].ulDensity = 100;
    rgindexcreateT[0].lcid = MAKELCID(0x409, SORT_DEFAULT);
    rgindexcreateT[0].cbVarSegMac = 0;
    rgindexcreateT[0].rgconditionalcolumn = NULL;
    rgindexcreateT[0].cConditionalColumn = 0;
    rgindexcreateT[0].err = JET_errSuccess;

    tablecreateT.cbStruct = sizeof(tablecreateT);
    tablecreateT.szTableName = (char *)szPivotTable;
    tablecreateT.szTemplateTableName = NULL;
    tablecreateT.ulPages = 16;
    tablecreateT.ulDensity = 80;
    tablecreateT.rgcolumncreate = rgcolumncreateT;
    tablecreateT.cColumns = 2;
    tablecreateT.rgindexcreate = rgindexcreateT;
    tablecreateT.cIndexes = 1;
    tablecreateT.grbit = NO_GRBIT;
    tablecreateT.tableid = JET_tableidNil;
    tablecreateT.cCreated = 0;

    Call(JetCreateTableColumnIndex(sesidT, dbidDatabase, &tablecreateT));
    Call(JetCloseTable(sesidT, tablecreateT.tableid));

HandleError:
    if (err < 0)
    {
        JetRollback(sesidT, 0);
        throw 0;
    }
}
