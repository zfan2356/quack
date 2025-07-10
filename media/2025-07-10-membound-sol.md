# Getting Memory-bound Kernels to Speed-of-Light
Wentao Guo, Ted Zadouri, Tri Dao

To make GPUs go brrr for both model training and inference, one has to optimize both compute-bound kernels (e.g. matmul, attention) and memory-bound kernels (pretty much everything else, such as elementwise, normalization, loss). [Matmul](https://docs.nvidia.com/cuda/cublas/) and [attention](https://github.com/Dao-AILab/flash-attention/tree/main) are already some of the most heavily optimized subroutines that we have. Here we instead focus on memory-bound kernels, where most of the time is spent on memory access (IO) instead of on actual computation. By understanding and exploiting the thread and memory hierarchy on modern accelerators, we can get these kernels to close to speed-of-light (as fast as theoretically possible). Thanks to the recent [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html), we can do so right in the comfort of an ergonomic Python environment, without having to touch CUDA C or C++.

For memory-bound kernels, the ratio between the number of Floating-point Operations (FLOPs) consumed and the number of bytes transferred is small (such ratio is called Arithmetic Intensity). Once a kernel’s arithmetic intensity enters the memory-bound regime, the kernel's throughput is determined by how many bytes/second the kernel can deliver rather than by FLOPs/second the kernel computes.

![Arithmetic intensity of a memory-bound softmax kernel is
O(1)](our-16k-131k-arithmetic-intensity-white.png "Arithmetic intensity of a memory-bound softmax kernel is O(1)")

Within the memory-bound kernels, elementwise activation is usually easier to deal with — it is inherently perfectly parallel as there are no dependencies across the elements. However, reduction operations are also prevalent in DL operators such as softmax and RMSNorm, and they require an aggregation of all values. A parallel associative reduction algorithm will execute O(log(#reduced dim)) rounds of partial reduction across threads in different spatiality where our knowledge of GPU memory hierarchy would help.

![Parallel maximum reduction](max_reduction.png "Parallel maximum reduction [1]")

In this blogpost, we describe how we can leverage the GPU memory hierarchy to implement efficient reduction kernels. As an example, we use CuTe DSL to implement 3 commonly used kernels in Large Language Models: RMSNorm, softmax, and cross entropy loss. We want to hit the maximum hardware throughput, or “GPU speed-of-light throughput”, and we need 2 ingredients: (1) global memory coalesce load/store (2) hardware-aware reduction strategy. As a bonus, we’ll also explain cluster reduction, a relatively new feature on Nvidia GPUs starting with Hopper (H100), and how that helps with very large reductions. This blogpost will explain the details of these ingredients and flesh out how they allow us to write speed-of-light kernels. Let’s start our journey!

<!-- ![Duck rocket](max_reduction.png "Parallel maximum reduction [1]") -->

Our code can be found at 🦆 <https://github.com/Dao-AILab/quack> 🦆
