/* Just a wrapper over apsdk. Please see the apsdk header files for documentation
   
   The need of the wrapper comes out the following facts:
 - apsdk is compiled with exceptions and including some of it's header file requires to
 have the Ols code compiled with exceptions
 - apsdk uses internaly STL which can throw exceptions. AP developers could not give guaranties
 that apsdk API does not throw exceptions. So we need to wrap everyting around try/catch
 - AP developers suggested to include only once apsdk.lib in the application. This is because
 apsdk uses some globals that should be initialized only once

*/


#pragma once

#include "Windows.h"
#include "TemplatedHResult.h"
#include <string>
#include <vector>

#ifdef ENABLE_APWRAPPER_API
#define APWRAPPER_API __declspec(dllexport)
#else
#define APWRAPPER_API __declspec(dllimport)
#endif

#ifdef _DEBUG
// User printf to validate log format string against the arguments
#define OSLOG(level, title, ...) (0 && printf(__VA_ARGS__), APWrapper::LogMessageV(__FILE__, __LINE__, __FUNCTION__, (level), (title), __VA_ARGS__))
#define OSLOG_ID(level, logid, title, ...) (0 && printf(__VA_ARGS__), APWrapper::LogMessageV(__FILE__, __LINE__, __FUNCTION__, (level), (logid), (title), __VA_ARGS__))
#define OSLOGASSERT(expr, ...) (0 && printf(__VA_ARGS__), APWrapper::LogAssertMessageV(__FILE__, __LINE__, __FUNCTION__, (expr), __VA_ARGS__))
#else
#define OSLOG(level, title, ...) APWrapper::LogMessageV(__FILE__, __LINE__, __FUNCTION__, (level), (title), __VA_ARGS__)
#define OSLOG_ID(level, logid, title, ...) APWrapper::LogMessageV(__FILE__, __LINE__, __FUNCTION__, (level), (logid), (title), __VA_ARGS__)
#define OSLOGASSERT(expr, ...) APWrapper::LogAssertMessageV(__FILE__, __LINE__, __FUNCTION__, (expr), __VA_ARGS__)
#endif

#define OSLOGDEBUG(title, ...) OSLOG(APWrapper::LogLevel_Debug, (title), __VA_ARGS__)
#define OSLOGINFO(title, ...) OSLOG(APWrapper::LogLevel_Info, (title), __VA_ARGS__)
#define OSLOGWARNING(title, ...) OSLOG(APWrapper::LogLevel_Warning, (title),  __VA_ARGS__)
#define OSLOGERROR(title, ...) OSLOG(APWrapper::LogLevel_Error, (title), __VA_ARGS__)

// When logid and title are combined into one macro, call the OSLOG... function without _ID.
#define OSLOGDEBUG_ID(logid, title, ...) OSLOG_ID(APWrapper::LogLevel_Debug, (logid), (title), __VA_ARGS__)
#define OSLOGINFO_ID(logid, title, ...) OSLOG_ID(APWrapper::LogLevel_Info, (logid), (title), __VA_ARGS__)
#define OSLOGWARNING_ID(logid, title, ...) OSLOG_ID(APWrapper::LogLevel_Warning, (logid), (title), __VA_ARGS__)
#define OSLOGERROR_ID(logid, title, ...) OSLOG_ID(APWrapper::LogLevel_Error, (logid), (title), __VA_ARGS__)

const DWORD c_counterMappedFileSize = 10 * 1024 * 1024;
static const char* RSL_BOOTSTRAP_WITH_MARKER = "RSL_BOOTSTRAP_WITH_MARKER";

extern "C"
{
    // C functions for public and managed use
    HRESULT APWRAPPER_API ApWrapperInitLibrary();
}

namespace apsdk
{
    class Uint64Counter;
    class Uint32Counter;
    class WatchdogOutput;
    class DMResponse;
    class DMClient;
    class APFileSync;
    class Checksum;
    struct CustomLogID;
    enum QuickCRCType : int;

    namespace configuration
    {
        class IConfiguration;
    }
}

namespace APWrapper
{

//This should succeed if
//  a.	Not running on AP 
//  b.	Running on non-AP but using AP client and AP client is setup correctly.
//  c.	Running on AP and AP is setup correctly.
//This should fail if 
//  a.	Running on Ap client and something wrong with AP client setup.
//  b.	Running on AP and something wrong with AP client setup.
HRESULT APWRAPPER_API __stdcall InitLibrary();


HRESULT APWRAPPER_API __stdcall InitLibraryA
(
    LPCSTR loggingRulesFileName
);

HRESULT APWRAPPER_API __stdcall InitLibraryB
(
    LPCSTR dataDir,
    LPCSTR networkDataDir,
    LPCSTR environment,
    LPCSTR cluster
);

HRESULT APWRAPPER_API SetConfigDir
(
    LPCSTR configDir
);

void APWRAPPER_API GetDefaultLoggingRulesConfigFile(LPSTR szBuff, int cbBuff);

BOOL APWRAPPER_API IsLibraryInitialized();

HRESULT APWRAPPER_API ClearInitializationAttemptCounters();

APWRAPPER_API const char * GetEnvironmentName();

APWRAPPER_API const char * GetSku();

APWRAPPER_API const char * GetMachineName();

APWRAPPER_API const char * GetMachineFunction();

APWRAPPER_API std::string GetVirtualEnvironment();

APWRAPPER_API HResult<LPSTR> GetConfiguredAutopilotIniPath();

APWRAPPER_API HResult<LPSTR> GetConfiguredAppFolderPath();

APWRAPPER_API HResult<LPSTR> GetDeploymentFolderPath();

APWRAPPER_API HResult<LPSTR> GetDeploymentFolderName();

APWRAPPER_API HResult<LPSTR> GetDataDirFolderPath();

APWRAPPER_API VOID ReleaseBuffer
(
    LPSTR string
);

struct APWRAPPER_API CustomLogId
{
    explicit CustomLogId(const char* name);
    const apsdk::CustomLogID& ApLogId() const;
    ~CustomLogId();
private:
    CustomLogId(const CustomLogId& another); // not implemented to prevent copying
    apsdk::CustomLogID* m_apLogId;
};

#define DEFINE_CUSTOM_LOGID(x) namespace {APWrapper::CustomLogId LogId_##x(#x);}

typedef enum
{
    LogLevel_Debug,
    LogLevel_Info,
    LogLevel_Status,
    LogLevel_Warning,
    LogLevel_Error,
    LogLevel_Assert
} APLogLevel;

VOID APWRAPPER_API LogMessage
(
    LPCSTR file,
    LPCSTR function,
    INT line,
    APLogLevel level,
    LPCSTR message
);

VOID APWRAPPER_API LogMessageA
(
    LPCSTR file,
    LPCSTR function,
    INT line,
    APLogLevel level,
    LPCSTR title,
    LPCSTR message
);

VOID APWRAPPER_API LogCoprocMessageA
(
    LPCSTR file,
    LPCSTR function,
    INT line,
    APLogLevel level,
    LPCSTR title,
    LPCSTR message
);

// wrapper to ap logs with variable args
VOID APWRAPPER_API LogMessageV
(
    LPCSTR file,
    INT line,
    LPCSTR function,
    APLogLevel level,
    LPCSTR title,
    _Printf_format_string_ LPCSTR szFormatString, 
    ...
);

VOID APWRAPPER_API LogMessageV2  // Cannot use the same name LogMessageV. Otherwise macro OSLOG resolves to this version incorrectly.
(
    LPCSTR file,
    INT line,
    LPCSTR function,
    APLogLevel level,
    LPCSTR title,
    _Printf_format_string_ LPCSTR szFormatString, 
    va_list argList
);

VOID APWRAPPER_API LogMessageV
(
    LPCSTR file,
    INT line,
    LPCSTR function,
    APLogLevel level,
    const CustomLogId& logId,
    LPCSTR title,
    _Printf_format_string_ LPCSTR szFormatString, 
    ...
);

VOID APWRAPPER_API LogMessageV2
(
    LPCSTR file,
    INT line,
    LPCSTR function,
    APLogLevel level,
    const CustomLogId& logId,
    LPCSTR title,
    _Printf_format_string_ LPCSTR szFormatString, 
    va_list argList
);

VOID APWRAPPER_API LogAssertMessageV
(
    LPCSTR file,
    INT line,
    LPCSTR function,
    bool expr,
    _Printf_format_string_ LPCSTR szFormatString, 
    ...
);

VOID APWRAPPER_API FlushLog
(
);

typedef enum
{
    CounterType_Uint32,
    CounterType_Uint64,
    CounterType_String
} CounterType;

const unsigned int c_flagPercentiles = 4;
typedef enum
{
    CounterFlag_None = 0,
    CounterFlag_Number = 1,
    CounterFlag_Rate = 2,
    CounterFlag_Number_Percentiles = CounterFlag_Number | c_flagPercentiles,
    CounterFlag_Rate_Percentiles = CounterFlag_Rate | c_flagPercentiles
} CounterFlags;

class APWRAPPER_API Uint64Counter
{
    apsdk::Uint64Counter *m_apsdkUint64Counter;

public:

    explicit Uint64Counter(apsdk::Uint64Counter* uint64Counter = NULL);

    // it would be better to have a kind of shared_ptr, but then wouldn't be able to avoid additional 'new' operation
    // so we just provide a function for explicit deletion of the internal pointer
    // it doesn't delete the object (this) itself
    void Uninitialize();

    BOOL IsInitialized() CONST;

    bool Set
    (
        unsigned __int64 val
    );
    bool Set
    (
        unsigned __int64 val,
        const char * instanceName
    );

    bool Increment
    (
    );
    bool Increment
    (
        const char * instanceName
    );

    bool Decrement
    (
    );
    bool Decrement
    (
        const char * instanceName
    );


    bool Add
    (
        unsigned __int64 val
    );
    bool Add
    (
        unsigned __int64 val,
        const char * instanceName
    );

    bool Subtract
    (
        unsigned __int64 val
    );
    bool Subtract
    (
        unsigned __int64 val,
        const char * instanceName
    );

    unsigned __int64 GetValue
    (
    );
    unsigned __int64 GetValue
    (
        const char * instanceName
    );

    unsigned __int64 GetPercentile
    (
        int position
    );
    unsigned __int64 GetPercentile
    (
        int position, 
        const char * instanceName
    );

};

class APWRAPPER_API Uint32Counter
{
    apsdk::Uint32Counter *m_apsdkUint32Counter;

public:

    explicit Uint32Counter(apsdk::Uint32Counter* uint32Counter = NULL);

    // it would be better to have a kind of shared_ptr, but then wouldn't be able to avoid additional 'new' operation
    // so we just provide a function for explicit deletion of the internal pointer
    // it doesn't delete the object (this) itself
    void Uninitialize();

    BOOL IsInitialized() CONST;

    bool Set
    (
        unsigned __int32 val
    );
    bool Set
    (
        unsigned __int32 val,
        const char * instanceName
    );

    bool Increment
    (
    );
    bool Increment
    (
        const char * instanceName
    );

    bool Decrement
    (
    );
    bool Decrement
    (
        const char * instanceName
    );


    bool Add
    (
        unsigned __int32 val
    );
    bool Add
    (
        unsigned __int32 val,
        const char * instanceName
    );

    bool Subtract
    (
        unsigned __int32 val
    );
    bool Subtract
    (
        unsigned __int32 val,
        const char * instanceName
    );

    unsigned __int32 GetValue
    (
    );
    unsigned __int32 GetValue
    (
        const char * instanceName
    );

    unsigned __int32 GetPercentile
    (
        int position
    );
    unsigned __int32 GetPercentile
    (
        int position, 
        const char * instanceName
    );

};

class APWRAPPER_API Counters
{
public:

    static Uint64Counter CreateUint64Counter
    (
        const char *section,
        const char *name,
        const CounterFlags flags
    );

    static Uint32Counter CreateUint32Counter
    (
        const char *section,
        const char *name,
        const CounterFlags flags
    );

    static DWORD GetMaxCountersFileSize();
};

bool APWRAPPER_API SetInt64Counter(const char *section, const char *name, const CounterFlags flags, unsigned __int64 val, const char * instanceName);
bool APWRAPPER_API IncrementInt64Counter(const char *section, const char *name, const CounterFlags flags, const char * instanceName);

class APWRAPPER_API IWatchdogPropertyManager
{

public:

    enum Levels
    {
        Level_Fatal,
        Level_Hardware,
        Level_Error,
        Level_Warning,
        Level_Ok,
        Level_Notify,
        Level_Alert,
        Level_Audit,
        Level_MAX
    };

    virtual IWatchdogPropertyManager::Levels SetProperty
    (
        char const * const szMachineName
      , IWatchdogPropertyManager::Levels Level
      , char const * const Property
      , char const * const Description
    ) = 0;
    
    virtual bool SendToDM
    (
        void
    ) = 0;

    virtual ~IWatchdogPropertyManager()
    {
    }
} ;



class APWRAPPER_API WatchdogPropertyManager : public IWatchdogPropertyManager
{
public :

    WatchdogPropertyManager();

    ~WatchdogPropertyManager();

    virtual IWatchdogPropertyManager::Levels SetProperty
    (
        char const * const szMachineName
      , IWatchdogPropertyManager::Levels Level
      , char const * const Property
      , char const * const Description
    );
    
    virtual bool SendToDM
    (
        void
    );

    virtual void Clear();

private:

    apsdk::WatchdogOutput* m_WatchdogOutput;

    WatchdogPropertyManager(CONST WatchdogPropertyManager&);
    WatchdogPropertyManager& operator=(CONST WatchdogPropertyManager&);
};

class DMClient;
class BuiltinDMResponseHandler;

class APWRAPPER_API OLSDMResponse
{
public:
    enum StatusCode
    {
        Ok,
        BadRequest,
        CommandNotFound,
        IncorrectUsage,
        UnAuthorized,
        DatabaseError,
        InternalError,
        StatusCount
    };

    static const char* StatusCodeToString(StatusCode statusCode);
    StatusCode Status() const;
    char* GetBuffer() const;
    int GetBufLength() const;
    LONG AddRef();
    LONG Release();
    LONG GetRefCount() const;

private:
    friend class DMClient;
    friend class BuiltinDMResponseHandler;

    OLSDMResponse();
    OLSDMResponse(apsdk::DMResponse* pDMResponse);
    ~OLSDMResponse();

    apsdk::DMResponse* m_pAPDMResponse;
    mutable volatile __declspec(align(4)) long m_refCnt;
};

class APWRAPPER_API OLSDMResponseHandler
{
public:
    //caller is responsible to call pResp->Release() upon finish using the response object.
    virtual void HandleDMResponse(void *ctx, bool success, OLSDMResponse *pResp) = 0;
};

class APWRAPPER_API DMClient
{
public:
    
     DMClient();
     ~DMClient();
    
    HRESULT Init(__in const char* szCluster = NULL,
                 __in OLSDMResponseHandler* pOlsDMResponseHandler = NULL,                 
                 __in DWORD sendTimeout = 300000, 
                 __in DWORD recvTimeout = 300000);

    //Caller should issue Release() upon finish using DMResponse
    HRESULT SendCommandSync(
            __in char const * const command,
            __in char const * const data,
            __in bool readOnly,
            __out OLSDMResponse** ppResponse);

    //From DM code, looks like ctx has to be no-null to get into async mode.
    HRESULT SendCommandAsync(
            __in const char* command,
            __in void* ctx,
            __in bool readOnly = true);
private:
    BuiltinDMResponseHandler* m_pDMResponseHandler;
    apsdk::DMClient* m_pClient;
};

class APWRAPPER_API BRSConfigReader
{
public:
    BRSConfigReader(){}
    ~BRSConfigReader(){}
    
    static HRESULT ReadKey(
        char const * const section, 
        char const * const key, 
        char const * const defaultValue, 
        char returnedValue[],
        const DWORD bufferSize,
        char const * const fileName,
        DWORD & returnedBufferSize
    );

    static HRESULT GetWathdogMaxAlertLevel(
        IWatchdogPropertyManager::Levels & Level
        );
};

class APWRAPPER_API FileSync
{
public:
    struct APWRAPPER_API Statistics
    {
        UINT64 transfBytesTotal;
    };

    FileSync();
    ~FileSync();
    bool AddManifest(const void *bufferStart, size_t bufferSize);
    bool Init(const char *root, const char *sourceRoot, bool verify, const char *tempManifest, bool noHttp = false, bool noSMB = false, bool ignoreCRC = false);
    const char* GetErrorMessage() const;
    void SetFullVerificationMode(bool value);
    void SetMaxBadwidth(ULONG32 bytesPerSecond);
    void SetResumeMode(bool value);
    void SetWriteBufferParams(const UINT32 bufferSize, const UINT32 bufferCount);
    bool SyncFiles(size_t* pnFilesCopied = NULL, volatile const bool* pStopFlag = NULL, FileSync::Statistics* pStat = NULL);
private:
    apsdk::APFileSync* m_pAPFileSync;
};

class APWRAPPER_API Checksum
{
public:
    Checksum();
    ~Checksum();

    void Init();
    void Init(ULONGLONG hash);
    void Init(const FILETIME &ft);
    bool Parse(const char* sHash, int hashSize, apsdk::QuickCRCType crcType);
    void AddData(const BYTE *bData, DWORD dwDataLen);
    void ToHexString(__out_ecount(hashSize) char *sHash, int hashSize) const;
    ULONGLONG ToULONGLONG() const;
    static bool CalculateFullCRC(const char *fileName, __out_ecount(hashSize) char *sHash, int hashSize, bool debugCRC = false,
                                 unsigned int uReadBufNum = 0, unsigned int uReadBufSize = 0, DWORD fileFlags = FILE_FLAG_NO_BUFFERING);
    static bool CalculateQuickCRC(const char *fileName, __out_ecount(hashSize) char *sHash, int hashSize, apsdk::QuickCRCType crcType,
                                  unsigned int uReadBufNum = 0, unsigned int uReadBufSize = 0, DWORD fileFlags = FILE_FLAG_NO_BUFFERING);
private:
    apsdk::Checksum* m_pChecksum;
};

class APWRAPPER_API APConfig
{
public:
    APConfig(const char* filePath);
    ~APConfig();
    bool IsLoaded();
    void GetSectionNames(std::vector<std::string>& sectionNames);
    void GetParameterNames(const char* sectionName, std::vector<std::string>& parameterNames);
    bool GetStringParameter(const char* sectionName, const char* parameterName, std::string& value);

private:
    const apsdk::configuration::IConfiguration* m_apConfig;
};

bool APWRAPPER_API DeleteDirectoryAndContents(const char* dirname, bool persistent = false, bool bDeleteReadOnlyFiles = false);
}

DEFINE_CUSTOM_LOGID(TSMetadataUpdate);
DEFINE_CUSTOM_LOGID(ReplStatus);
DEFINE_CUSTOM_LOGID(ClientStatus);
DEFINE_CUSTOM_LOGID(CosmosBackup);
DEFINE_CUSTOM_LOGID(OLSIndexQuery);
DEFINE_CUSTOM_LOGID(CosmosClient);
DEFINE_CUSTOM_LOGID(IngestionAPIRequest);
DEFINE_CUSTOM_LOGID(DiskPrioritization);
DEFINE_CUSTOM_LOGID(ServerStateUpdate);
DEFINE_CUSTOM_LOGID(IngestionWorkerRequest);
DEFINE_CUSTOM_LOGID(P2P);
