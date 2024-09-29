def get_workload_info(workload : str) -> float:
    table = {
        "megatron-126M": {
            "ai": 1,
            "size_GB": 0 
        },
        "megatron-5B": {
            "ai": 1,
            "size_GB": 0 
        },
        "megatron-22B": {
            "ai": 1,
            "size_GB": 0 
        },
        "megatron-40B": {
            "ai": 1,
            "size_GB": 0 
        },
        "megatron-1T": {
            "ai": 1,
            "size_GB": 0 
        },
        "gpt3-13B": {
            "ai": 1,
            "size_GB": 0 
        },
        "gpt3-175B": {
            "ai": 1,
            "size_GB": 0 
        },
        "lambda": {
            "ai": 1,
            "size_GB": 0 
        },
    }
    assert(workload in table), f"Workload {workload} not in table."
    return table[workload]

def get_mem_info(mem_type : str) -> dict:
    """ param per memory unit
    """
    table = {
        "HBM3": {
            "bw_GBps": 1,
            "lat_ns": 0,
            "cap_GB": 1
        },
        "HBM4": {
            "bw_GBps": 1,
            "lat_ns": 0,
            "cap_GB": 1
        },
        "DDR5": {
            "bw_GBps": 1,
            "lat_ns": 0,
            "cap_GB": 1
        }
    }
    assert(mem_type in table), f"Memory type {mem_type} not in table."
    return table[mem_type]

def get_cu_info(gpu : str, datatype : str) -> dict:
    """ all units in FLOPs
    """
    table = {
        "h100": {
            "float8": {
                "matrix": 2000e12,
                "vector": 120e12
            },
            "float16": {
                "matrix": 1000e12,
                "vector": 120e12
            }
        },
        "a100": {
            "float16": {
                "matrix": 312e12,
                "vector": 78e12
            }
        }
    }
    assert(gpu in table), f"GPU {gpu} not in table"
    assert(datatype in table[gpu]), f"Datatype {datatype} not in table for {gpu}"
    return table[gpu][datatype]
    