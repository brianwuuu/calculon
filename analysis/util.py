import json, itertools, os

# Parse JSON file into dictionary object
def parseJSON(filename):
    print("[Analysis] Parsing JSON file " + filename)
    assert(os.path.isfile(filename)), filename
    with open(filename) as json_file: 
        json_dict = json.load(json_file)
    return json_dict

def dumpJSON(filename, map):
    with open(filename, "w+") as json_file:
        json.dump(map, json_file)
        
# Convert memory string to float with unit "Byte"
def toBytes(mem_str):
    val, unit = mem_str.split(" ")
    factor = 1
    if unit == "TiB": factor = 10 ** 12
    elif unit == "GiB": factor = 10 ** 9
    elif unit == "MiB": factor = 10 ** 6
    elif unit == "KiB": factor = 10 ** 3
    elif unit == "B": factor = 1
    else: raise Exception("[Error] Invalid memory unit: {}".format(unit))
    return float(val) * factor
        
def generateSystemFileNameString(sys_config):
    compute_str = "{}tflops_".format(sys_config["matrix"]["float16"]["tflops"])
    compute_str += "rl_" if sys_config["processing_mode"] == "roofline" else ""
    mem_str = "mem1_{}GBps_{}GB_mem2_{}GBps_{}GB_".format(
        sys_config["mem1"]["GBps"], sys_config["mem1"]["GiB"],
        sys_config["mem2"]["GBps"], sys_config["mem2"]["GiB"],
    )
    network_str = "net1_{}GBps_{}s_{}pu_net2_{}GBps_{}s_{}pu".format(
        sys_config["networks"][0]["bandwidth"], sys_config["networks"][0]["latency"], sys_config["networks"][0]["processor_usage"],
        sys_config["networks"][1]["bandwidth"], sys_config["networks"][1]["latency"], sys_config["networks"][1]["processor_usage"]
    )
    system_filename = compute_str + mem_str + network_str + ".json"
    return system_filename

def generateArchFileNameString(arch_config):
    str_builder = "{}_t{}_p{}_d{}_mbs{}{}{}{}_full".format(
        arch_config["num_procs"],
        arch_config["tensor_par"], arch_config["pipeline_par"], arch_config["data_par"],
        arch_config["microbatch_size"],
        "_wo" if arch_config["weight_offload"] == True else "",
        "_ao" if arch_config["activations_offload"] == True else "",
        "_oo" if arch_config["optimizer_offload"] == True else ""
    )
    return str_builder

def generateModelFileNameString(model_config):
    str_builder = "{}h_{}ff_{}ss_{}ah_{}as_{}nb".format(
        model_config["hidden"], model_config["feedforward"], model_config["seq_size"], 
        model_config["attn_heads"], model_config["attn_size"], model_config["num_blocks"])
    return str_builder

def extractTimingInfo(stats):
    timing = {}
    timing["FW Pass"] = stats["Batch FW time"]
    timing["BW Pass"] = stats["Batch BW time"]
    timing["Optim Step"] = stats["Batch optim time"]
    timing["PP Bubble"] = stats["Batch bubble overhead"]
    timing["FW Recompute"] = stats["Batch recompute overhead"] + stats["Batch recomm overhead"]
    timing["TP Comm"] = stats["Batch TP comm time on link"]
    timing["PP Comm"] = stats["Batch PP comm time on link"]
    timing["DP Comm"] = stats["Batch DP comm time on link"]
    return timing

def extractMemoryInfo(stats):
    usage = {}
    usage["Weights"] = stats["Weights"]
    usage["Activations"] = stats["Act"] + stats["Act CP"]
    usage["Act Gradients"] = stats["Act grad"]
    usage["Weight Gradients"] = stats["Weight grad"]
    usage["Optimizer Space"] = stats["Optim space"]
    return usage

def cartesianProduct(param_list):
    return [x for x in itertools.product(*param_list)]