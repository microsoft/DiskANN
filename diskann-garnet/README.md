# diskann-garnet

This crate provides an implementation of `DataProvider` for
[Garnet](https://github.com/microsoft/garnet) as well as FFI endpoints for
Garnet to access DiskANN functionality. Garnet is a remote cache service
developed by Microsoft Research with Redis compatibility, and has better
performance, throughput, and lower latency than competitors. With this crate, it
also supports vector sets, allowing clients to use vector sets for ANN indexing
and search.

## Supported Features

diskann-garnet currently supports vectors with element types of 32-bit float and 8-bit signed or unsigned integers. Indexes can be full precision or quantized. When the index is quantized, the full precision vectors are still stored in order to rerank the final candidates during search operations, which improves recall.

In addition to the normal vector set operations, the following extensions are
added:

### New Element Types

- `XI8`: signed 8-bit integers
- `XU8`: unsigned 8-bit integers

### Distance Metrics

Redis always uses cosine distance, but many vector data sets use other metrics. The following metrics can be used by passing `XDISTANCE_METRIC <metric>` options to the `VADD` command. The default is `L2`.

- `COSINE`
- `XCOSINE_NORMALIZED`
- `IP` (inner product)
- `L2` (euclidean)

### New Quantizers

- `XNOQUANT_I8`: full precision 8-bit signed integer
- `XNOQUANT_U8`: full precision 8-bit unsigned integer
- `XBIN_I8`: binary quantization of 8-bit signed integer (using DiskANN's spherical quantizer based on RaBitQ)
- `XBIN_U8`: binary quantization of 8-bit unsigned integer (using DiskANN's spherical quantizer based on RaBitQ)


Currently there is a limit of `2^32 - 1` vectors in a single instance due to
internal IDs being `u32`. This restriction will be lifted in the future.

## Installing

Garnet depends on diskann-garnet as a NuGet package, which means you can simply
check out the Garnet repo on Windows or Linux, and if you have a dotnet
toolchain installed you can just run:

```sh
dotnet run -c Release -f net10.0 --project main/GarnetServer -- --enable-vector-set-preview
```

and it will build and launch Garnet with vector sets enabled.

### Local Installs

If you want to install a specific version of diskann-garnet to use with Garnet,
it is a little more complicated. Aside from compiling diskann-garnet, you will
need to create a NuGet package. For example:

```pwsh
cd diskann-garnet
cargo build --release
mkdir ../target/pkg
mkdir ../target/pkg/linux
mkdir ../target/pkg/windows
mkdir ../target/pkg/docs
cp diskann-garnet.nuspec ../target/pkg
cp README.md ../target/pkg/linux/libdiskann_garnet.so # dummy file
cp ../target/release/*.dll ../target/pkg/windows
cp ../target/release/*.pdb ../target/pkg/windows
cp README.md ../target/pkg/docs
nuget pack -BasePath ../target/pkg -OutputDirectory LOCAL_NUGET_PATH
nuget locals all -clear
```

You will need to set up a local path to host NuGets and setup
`%APPDATA%/NuGet/NuGet.config` appropriately. For example:

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <add key="local" value="LOCAL_NUGET_PATH" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" protocolVersion="3" />
  </packageSources>
  <packageSourceMapping>
    <packageSource key="nuget.org">
        <package pattern="*"/>
    </packageSource>
    <packageSource key="local">
        <package pattern="diskann-garnet"/>
    </packageSource> 
  </packageSourceMapping>
</configuration>
```

Replace `LOCAL_NUGET_PATH` with whatever path you like.

Linux instructions are a bit more difficult as `nuget pack` does not exist in
Linux. You will need to grab an existing NuGet from NuGet.org, unzip it, and
then replace the files, and rezip.

```
mkdir target/nupkg
cd target/nupkg
unzip PATH_TO/diskann-garnet.x.y.z.nupkg
cd ../..
cargo build --release --package diskann-garnet
cp diskann-garnet/diskann-garnet.nuspec target/nupkg/
cp target/release/libdiskann_garnet.so target/nupkg/runtimes/linux-x64/native/
cd target/nupkg
zip -r LOCAL_NUGET_PATH/diskann-garnet.X.Y.Z.nupkg *
dotnet nuget locals all --clear
```

Replace `LOCAL_NUGET_PATH` with the path you like and `X.Y.Z` with the version
number from `diskann-garnet.nuspec`.

If you aren't replacing the same version of diskann-garnet as Garnet is using,
you can modify Garnet's `Directory.Packages.props` file to set the version to
the one you want.

## Testing

Unit tests are run in the usual way with `cargo test`, but many are end-to-end
and run from the Garnet side. These two invocations will run the relevant tests:

```
dotnet test test/standalone/Garnet.test.vectorset -f net10.0 -c Debug --filter RespVectorSetTests
dotnet test test/standalone/Garnet.test.extensions -f net10.0 -c Debug --filter DiskANNServiceTests
```

## Client Examples

To benchmark or see an example of usage, see the `vectorset` crate, which uses
the official Redis Rust client to run vector workloads on Garnet.