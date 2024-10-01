def get_workload_info(workload : str) -> float:
    table = {
        "megatron-126M": {
            "ai": 58.74,
            "size_GB": 4.248 
        },
        "megatron-5B": {
            "ai": 416.60,
            "size_GB": 80.631  
        },
        "megatron-22B": {
            "ai": 486.66,
            "size_GB": 341.986 
        },
        "megatron-40B": {
            "ai": 701.971,
            "size_GB": 597.023 
        },
        "megatron-1T": {
            "ai": 1492.673,
            "size_GB": 14747 
        },
        "gpt3-13B": {
            "ai": 497.19,
            "size_GB": 200.905 
        },
        "gpt3-175B": {
            "ai": 924.698,
            "size_GB": 2572 
        },
        "lamda": {
            "ai": 755.395,
            "size_GB": 1536 
        },
        "anthropic-52B": {
            "ai": 351.32,
            "size_GB": 975.531
        },
        "chinchilla": {
            "ai": 701.971,
            "size_GB": 985.039
        },
        "palm-540B": {
            "ai": 1493.166,
            "size_GB": 6276
        },
        "turing-530B": {
            "ai": 1342.950,
            "size_GB": 7760
        }
    }
    assert(workload in table), f"Workload {workload} not in table."
    return table[workload]

def get_mem_info(mem_type : str) -> dict:
    """ param per memory unit
    OMI: https://blocksandfiles.com/2021/05/11/omi_serial_bus_white_paper/
    CXL: A Case for CXL-Centric Server Processors
    DDR4: https://en.wikipedia.org/wiki/DDR4_SDRAM
    DDR5: https://en.wikipedia.org/wiki/DDR5_SDRAM
    """
    table = {
        "HBM3": {
            "bw_GBps": 800,
            "lat_ns": 106.7,
            "cap_GB": 24
        },
        "HBM4": {
            "bw_GBps": 1400,
            "lat_ns": 0,
            "cap_GB": 32
        },
        "DDR4": {
            "bw_GBps": 25.6,
            "lat_ns": 73.3,
            "cap_GB": 64
        },
        "DDR5": {
            "bw_GBps": 64,
            "lat_ns": 73.3,
            "cap_GB": 512
        },
        "CXL": {
            "bw_GBps": 128,
            "lat_ns": 30+73.3,
            "cap_GB": 512
        }
    }
    assert(mem_type in table), f"Memory type {mem_type} not in table."
    return table[mem_type]

def get_cu_info(gpu : str, datatype : str) -> dict:
    """ all units in FLOPs
    H100: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
    B100: https://www.hyperstack.cloud/blog/thought-leadership/everything-you-need-to-know-about-the-nvidia-blackwell-gpus
          https://nvdam.widen.net/s/q8f9llv72p/nvidia-blackwell-architecture-technical-brief
    """
    table = {
        "b100": {
            "float8": {
                "matrix": 7e15,
                "vector": 120e12
            },
            "float16": {
                "matrix": 3.5e15,
                "vector": 120e12
            }
        },
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