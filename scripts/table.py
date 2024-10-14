def get_workload_info(workload : str) -> float:
    
    table = {
        "megatron-126M": {
            "training": {
                "ai": 58.74,
                "size_GB": 4.248
            },
            "inference": {
                "ai": 41.634,
                "size_GB": 1.68
            }
        },
        "megatron-530M": {
            "training": {
                "ai": 136.65,
                "size_GB": 9.59
            },
            "inference": {
                "ai": 105.55,
                "size_GB": 2.37
            }
        },
        "megatron-1B": {
            "training": {
                "ai": 196.519,
                "size_GB": 16.29
            },
            "inference": {
                "ai": 163.182,
                "size_GB": 3.306
            }
        },
        "megatron-5B": {
            "training": {
                "ai": 416.60,
                "size_GB": 80.631
            },
            "inference": {
                "ai": 344.018,
                "size_GB": 13.2
            }
        },
        "megatron-22B": {
            "training": {
                "ai": 404.402,
                "size_GB": 341.986
            },
            "inference": {
                "ai": 386.66,
                "size_GB": 50.166
            }
        },
        "megatron-40B": {
            "training": {
                "ai": 701.971, # 213.489,
                "size_GB": 640 # 597.023
            },
            "inference": {
                "ai": 636.488, # 136.489,
                "size_GB": 80 # 74.420
            }
        },
        "anthropic-52B": {
            "training": {
                "ai": 351.32,
                "size_GB": 975.531
            },
            "inference": {
                "ai": 247.64,
                "size_GB": 144.42
            }
        },
        "chinchilla-64B": {
            "training": {
                "ai": 701.971,
                "size_GB": 985.039
            },
            "inference": {
                "ai": 236.488,
                "size_GB": 135.968
            }
        },
        "gpt3-175B": {
            "training": {
                "ai": 840.82,
                "size_GB": 2427.9
            },
            "inference": {
                "ai": 802.26,
                "size_GB": 398.58
            }
        },
        "megatron-1T": {
            "training": {
                "ai": 1492.673,
                "size_GB": 14747 
            }
        },
        "gpt3-13B": {
            "training": {
                "ai": 497.19,
                "size_GB": 215.72 
            }
        },
    }
    assert(workload in table), f"Workload {workload} not in table."
    return table[workload]

def get_mem_info(mem_type : str) -> dict:
    """ param per memory unit
    HBM: https://en.wikipedia.org/wiki/High_Bandwidth_Memory
    OMI: https://blocksandfiles.com/2021/05/11/omi_serial_bus_white_paper/
    CXL: A Case for CXL-Centric Server Processors
    DDR4: https://en.wikipedia.org/wiki/DDR4_SDRAM
    DDR5: https://en.wikipedia.org/wiki/DDR5_SDRAM
    """
    table = {
        "HBM2": {
            "bw_GBps": 307, 
            "lat_ns": 106.7, # 106.7
            "cap_GB": 8
        },
        "HBM2E": { # H100 original
            "bw_GBps": 600, 
            "lat_ns": 106.7, # 106.7
            "cap_GB": 16
        },
        "HBM3": {
            "bw_GBps": 800,
            "lat_ns": 106.7,
            "cap_GB": 24
        },
        "HBM4": {
            "bw_GBps": 1400,
            "lat_ns": 106.7,
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
            "lat_ns": 73.3,
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