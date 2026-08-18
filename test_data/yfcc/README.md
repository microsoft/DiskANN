# YFCC Test Dataset

The YFCC test dataset is a 10000 size subset of the YFCC dataset released for the [NeurIPS'23 Big ANN Benchmarks competition](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/dataset_preparation/yfcc_filtered_dataset.md). It consists of embedded images released under a [variety of CC licenses](https://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/), and each embedding and metadata point takes on the license of the original image. The query set was later augmented with some additional metadata, and this metadata is used for filtered search. There are 100 queries in this small version of the dataset.

The dataset was originally `uint8`, but it is converted to `float32` for the test dataset. The dataset is Euclidean, but we compute groundtruth with respect to the cosine and inner product metrics for use in testing. 

The filters are curated to have varying match rates. The match rate statistics are as follows:
    average: 0.051061
    median:  0.008700
    min:     0.000000
    max:     0.339700

There is a streaming runbook provided. It tests searching, inserting, deleting, and replacing, and it is designed so that slots must be recycled.