# diskann-garnet

This crate providers an implementation of `DataProvider` for
[Garnet](https://github.com/microsoft/garnet) as well as FFI endpoints for
Garnet to access DiskANN functionality. Garnet is a remote cache service
developed by Microsoft Research, offers Redis compatibility, and has better
performance, throughput, and lower latency than competitors. With this crate, it
also supports vector sets, allowing clients to use vector sets for ANN indexing
and search.

## Supported Features

diskann-garnet currently supports full precision vectors only with cosine
distance metrics.

In addition to the normal vector set operations, the following extensions are
added:

- `XB8`: When specifying vector input type, you can use `XB8` instead of `FP32`
  to specify binary data in uint8 format, one byte per dimension.
- `XPREQ8`: This is a pseudo-quantizer that specifies the vector data will be
  stored as full precision data in uint8 format.

Generally you will use `XB8` with `XPREQ8` to input and store uint8 vectors and
`FP32` with `NOQUANT` to input and store f32 vectors.

Support for binary and scalar quantization is coming, along with support for
customizing the distance metric.

Currently there is limit of `2^32 - 1` vectors in a single instance due to
internal IDs being `u32`. This will probably restriction will be lifted in the
future.

## Installing

Garnet depends on diskann-garnet as a NuGet package, which means you can simply
check out the Garnet repo on Windows or Linux, and if you have a dotnet
toolchain installed you can just run:

```sh
dotnet dotnet run -c Release -f net8.0 --project main/GarnetServer --enable-vector-set-preview
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
cp README.md ../target/pkg/linux/libdiskann_garnet.so # dummy file
cp ../target/release/*.dll ../target/pkg/windows
cp ../target/release/*.pdb ../target/pkg/windows
cp README.md ../target/pkg/docs
nuget pack -BasePath ../target/pkg -OutputDirectory LOCAL_NUGET_PATH
nuget locals -clear all
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
cd ../../diskann-garnet
cargo build --release
cp diskann-garnet.nuspec ../target/nupkg/
cp ../target/release/libdiskann_garnet.so ../target/nupkg/runtimes/linux-x64/native/
cd ../target/nupkg
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
and run from the Garnet side. These two invocations will the relevant tests:

```
dotnet test test/Garnet.test -f net8.0 -c Debug --filter RespVectorSetTests
dotnet test test/Garnet.test -f net8.0 -c Debug --filter DiskANNServiceTests
```

## Client Examples

To benchmark or see an example of usage, see the `vectorset` crate, which uses
the official Redis Rust client to run vector workloads on Garnet.