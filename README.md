# Greedy Correlation Clustering
 
![Example clustering](https://github.com/CheezBarger/correlation-clustering/blob/master/examples/example_clustering.png)

This is an example implementation of correlation clustering through greedy VOTE and Best One Element Move (LOCALSEARCH) as provided by Elsner and Schudy (2009) and Gionis et al. (2007). The implementation is intended to cluster geographical areas based on Pearson's correlation between two pixels.

`[1] Elsner, M. and Schudy, W., “Bounding and Comparing Methods for Correlation Clustering Beyond ILP”, Proceedings ofthe NAACL HLT Workshop on Integer Linear Programming for Natural Language Processing, 2009`

`[2] Gionis, A., Mannila, H., and Tsaparas, P., “Clustering Aggregation”, ACM Transactions on Knowledge Discovery from Data, 2007`

This was originally written for a university project therefore I make little attempt to optimize it. The repository is thus only intended to provide reference, as it appears to be limited.

To run the example clustering use the following command:

```
    cargo run --release --example main
```
