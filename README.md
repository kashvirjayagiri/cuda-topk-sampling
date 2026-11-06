# cuda-topk-sampling

This project explores how GPU parallel algorithms can accelerate Top-k token selection during LLM inference.

We implement and compare different CUDA-based Top-k selection methods against a full-sort baseline and existing optimized implementations. Our goal is to evaluate their correctness, performance, and scalability across different workloads.

Rather than developing a novel algorithm, we focus on understanding the performance trade-offs between parallel selection strategies and identifying when each approach is most effective.
