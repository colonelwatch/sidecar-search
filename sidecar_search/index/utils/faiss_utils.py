import faiss


def to_gpu(index: faiss.Index, device: int = 0) -> faiss.Index:
    opts = faiss.GpuClonerOptions()
    opts.useFloat16 = True  # float16 is necessary for codes sized 56 bits and over
    env = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(env, device, index, opts)


def to_cpu(index: faiss.Index) -> faiss.Index:
    return faiss.index_gpu_to_cpu(index)
