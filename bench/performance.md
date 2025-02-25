## Command to run

```bash
mix run bench/vettore_bench.exs
```

## My PC Results (Intel(R) Core(TM) Ultra 5 125H)

```bash
Operating System: Linux
CPU Information: Intel(R) Core(TM) Ultra 5 125H
Number of Available Cores: 18
Available memory: 15.29 GB
Elixir 1.18.2
Erlang 27.2.2
JIT enabled: true

Benchmark suite executing with the following configuration:
warmup: 2 s
time: 3 s
memory time: 0 ns
reduction time: 0 ns
parallel: 1
inputs: none specified
Estimated total run time: 1 min 15 s

Benchmarking batch_insert_1000_binary ...
Benchmarking batch_insert_1000_cosine ...
Benchmarking batch_insert_1000_dot ...
Benchmarking batch_insert_1000_euclidean ...
Benchmarking batch_insert_1000_hnsw ...
Benchmarking similarity_search_binary ...
Benchmarking similarity_search_cosine ...
Benchmarking similarity_search_dot ...
Benchmarking similarity_search_euclidean ...
Benchmarking similarity_search_hnsw ...
Benchmarking single_insert_binary ...
Benchmarking single_insert_cosine ...
Benchmarking single_insert_dot ...
Benchmarking single_insert_euclidean ...
Benchmarking single_insert_hnsw ...
Calculating statistics...
Formatting results...

Name                                  ips        average  deviation         median         99th %
single_insert_hnsw               620.45 K        1.61 μs  ±1984.02%        1.37 μs        2.82 μs
single_insert_dot                585.92 K        1.71 μs  ±2093.49%        1.38 μs        2.89 μs
single_insert_binary             562.06 K        1.78 μs  ±2378.59%        1.36 μs        2.67 μs
single_insert_euclidean          534.20 K        1.87 μs  ±1710.27%        1.92 μs        3.06 μs
single_insert_cosine             503.73 K        1.99 μs  ±2263.61%        1.91 μs        2.77 μs
similarity_search_binary          16.43 K       60.85 μs    ±46.42%       59.76 μs      150.58 μs
similarity_search_euclidean       13.91 K       71.88 μs    ±48.66%       73.92 μs      139.96 μs
similarity_search_dot             13.57 K       73.68 μs    ±23.90%       74.43 μs      120.25 μs
similarity_search_cosine          11.86 K       84.29 μs    ±33.05%       82.12 μs      172.90 μs
similarity_search_hnsw            10.25 K       97.57 μs    ±41.99%       93.62 μs      198.98 μs
batch_insert_1000_hnsw             3.87 K      258.28 μs    ±18.43%      262.59 μs      377.67 μs
batch_insert_1000_euclidean        3.83 K      260.94 μs    ±38.08%      249.53 μs      599.66 μs
batch_insert_1000_binary           3.73 K      267.74 μs    ±19.36%      266.81 μs      447.58 μs
batch_insert_1000_dot              3.59 K      278.87 μs    ±35.56%      258.45 μs      650.09 μs
batch_insert_1000_cosine           3.53 K      282.92 μs    ±34.05%      307.24 μs      586.30 μs

Comparison: 
single_insert_hnsw               620.45 K
single_insert_dot                585.92 K - 1.06x slower +0.0950 μs
single_insert_binary             562.06 K - 1.10x slower +0.167 μs
single_insert_euclidean          534.20 K - 1.16x slower +0.26 μs
single_insert_cosine             503.73 K - 1.23x slower +0.37 μs
similarity_search_binary          16.43 K - 37.76x slower +59.24 μs
similarity_search_euclidean       13.91 K - 44.60x slower +70.27 μs
similarity_search_dot             13.57 K - 45.71x slower +72.07 μs
similarity_search_cosine          11.86 K - 52.30x slower +82.68 μs
similarity_search_hnsw            10.25 K - 60.54x slower +95.96 μs
batch_insert_1000_hnsw             3.87 K - 160.25x slower +256.67 μs
batch_insert_1000_euclidean        3.83 K - 161.90x slower +259.33 μs
batch_insert_1000_binary           3.73 K - 166.12x slower +266.13 μs
batch_insert_1000_dot              3.59 K - 173.02x slower +277.26 μs
batch_insert_1000_cosine           3.53 K - 175.54x slower +281.31 μs
```
## 1. Single Insert Benchmarks

**Single insert** across all distance metrics (HNSW, dot, binary, euclidean, cosine) are all in the range of **1.6–2.0 μs**. The differences here are minor:

- **HNSW** was unexpectedly the quickest for single insert (1.61 μs)  
- Then **dot** (1.71 μs), **binary** (1.78 μs), **euclidean** (1.87 μs), **cosine** (1.99 μs).

### Why So Similar?  
For small dimension (3D) and a small number of embeddings, the overhead of checking dimension + metadata + pushing to an internal `Vec` is likely dominating. HNSW is also quick for single insert because the data set is tiny, so updating the HNSW graph is relatively cheap.  

---

## 2. Batch Insert (1000 Embeddings)

Times are around **260–280 μs** for all metrics, which is extremely fast to insert 1000 embeddings. Differences are small:

| Metric       | Average time  |
|--------------|---------------|
| HNSW         | ~258.28 μs    |
| Euclidean    | ~260.94 μs    |
| Binary       | ~267.74 μs    |
| Dot          | ~278.87 μs    |
| Cosine       | ~282.92 μs    |

Again, for **3D** vectors and only 1000 embeddings, the overhead is quite low. Surprising to see HNSW top the list, but with so few embeddings, building an incremental small graph is cheap. In real large-scale usage, HNSW might trade off more memory for bigger data sets but scale better in searches.

---

## 3. Similarity Search

You tested searching for the **top‑k** (k=10) out of 1000 embeddings. The results (in microseconds) were:

| Metric     | Time   | Observed IPS |
|------------|--------|--------------|
| **Binary**      | ~60.85 μs  | 16.43 K    |
| **Euclidean**   | ~71.88 μs  | 13.91 K    |
| **Dot**         | ~73.68 μs  | 13.57 K    |
| **Cosine**      | ~84.29 μs  | 11.86 K    |
| **HNSW**        | ~97.57 μs  | 10.25 K    |

### Observations

1. **Binary** was the fastest search.  
   - Hamming distance is just bitwise XOR + popcount, which is very fast especially for small vectors (3D is basically a few bits).

2. **Euclidean/Dot/Cosine** are next, all fairly close.  

3. **HNSW** is slowest **at this tiny scale**.  
   - HNSW typically shines with bigger data sets, because it can skip a large portion of the embeddings. At 1000 embeddings, a naive full scan can actually be faster than building/traversing a hierarchical graph.

For bigger dimension (e.g., 128D, 512D) or bigger data sets (50k–1M embeddings), you might see HNSW start to outperform direct pairwise scans. Right now, the overhead of index traversal is bigger than the cost of scanning a small dataset.

---

## Conclusions

1. **For tiny data** (3D, 1k embeddings), **all inserts** are under a microseconds scale or a couple hundred microseconds for batch. They’re basically near each other.  
2. **Search** at this scale is also very quick for direct metrics. HNSW is slower because the indexing overhead is not amortized.  
3. If you tested **larger dimensions** (say 128D) or **larger data** (like 100k or 1M embeddings), you’d likely see bigger differences:
   - HNSW might become significantly faster at searching.  
   - Dot vs Euclidean vs Binary might shift in overhead, especially for large vectors.  

At a small scale, these differences are overshadowed by overheads of function calls, iteration, or index management. 


- **Single Insert**: ~1–2 μs each, nearly identical across metrics.  
- **Batch Insert (1000)**: ~250–280 μs, HNSW ironically a bit faster, but all close.  
- **Similarity Search**: **Binary** is fastest, then Euclidean, Dot, Cosine, and last HNSW.  
- **Reason**: With only 1000 embeddings & 3D vectors, a naive linear scan is extremely fast, overshadowing the benefits of advanced structures like HNSW. HNSW typically helps more with *larger* data sets and higher dimensions.