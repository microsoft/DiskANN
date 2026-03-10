using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DiskANNWrapper
{
    public class Index
    {
        const string _keywordsFileName = "_keywords.txt";
        private IntPtr _nativeIndexPtr = IntPtr.Zero;
        private string[] _keywords;
        private UInt32 _embeddingDim;
        private UInt32 _searchL;
        private bool _isInitialized = false;

        public bool Initialize(string indexPath, UInt32 embeddingDim = 64, UInt32 numThreads = 32, UInt32 searchL = 150)
        {
            if (_isInitialized || _nativeIndexPtr != IntPtr.Zero)
            {
                throw new Exception($"Index {indexPath} is already initialized.");
            }

            var keywordsFilePath = indexPath + _keywordsFileName;
            _keywords = System.IO.File.ReadAllLines(keywordsFilePath);
            Console.WriteLine($"Loaded {_keywords.Length} keywords from {keywordsFilePath}");

            _embeddingDim = embeddingDim;
            _searchL = searchL;
            _nativeIndexPtr = NativeMethods.CreateIndex(indexPath, embeddingDim, numThreads, searchL);
            _isInitialized = (_nativeIndexPtr != IntPtr.Zero);

            Console.WriteLine($"Index {indexPath} initialization status: {_isInitialized}");

            return _isInitialized;
        }

        public List<string> Search(byte[] embedding, uint topK, float distanceThreshold)
        {
            if (!_isInitialized)
            {
                throw new Exception("Index is not initialized.");
            }

            if (topK > _searchL)
            {
                throw new ArgumentException($"topK ({topK}) cannot be greater than searchL ({_searchL}).");
            }

            if (embedding == null)
            {
                throw new ArgumentNullException(nameof(embedding), "Embedding cannot be null.");
            }

            if (embedding.Length != _embeddingDim)
            {
                throw new ArgumentException($"Embedding dimension ({embedding.Length}) does not match index embedding dimension ({_embeddingDim}).");
            }

            if (distanceThreshold < 0)
            {
                throw new ArgumentException("Distance threshold cannot be negative.");
            }

            uint[] resultIds = new uint[topK];
            float[] distances = Enumerable.Repeat(-1f, (int)topK).ToArray();
            if (NativeMethods.SearchIndex(_nativeIndexPtr, embedding, topK, _searchL, resultIds, distances) == 0)
            {
                return resultIds.Where((id, index) => distances[index] < distanceThreshold && distances[index] >= 0 && id < _keywords.Length)
                    .Select(id => _keywords[id]).ToList();
            }

            return new List<string>();
        }

        ~Index()
        {
            if (_nativeIndexPtr != IntPtr.Zero)
            {
                NativeMethods.ReleaseIndex(_nativeIndexPtr);
                _nativeIndexPtr = IntPtr.Zero;
            }
        }
    }

    internal class NativeMethods
    {
        [DllImport("diskann.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr CreateIndex(
            [MarshalAs(UnmanagedType.LPStr)] string indexPath,
            uint embeddingDim,
            uint numThreads,
            uint searchL);

        [DllImport("diskann.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ReleaseIndex(IntPtr indexPtr);

        [DllImport("diskann.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int SearchIndex(
            IntPtr indexPtr,
            [In] byte[] query,
            uint recallAt,
            uint searchL,
            [Out] uint[] resultIds,
            [Out] float[] distances);
    }
}
