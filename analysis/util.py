import os, sys, stat, json, itertools

# Parse JSON file into dictionary object
def parseJSON(filename):
    with open(filename) as json_file: 
        json_dict = json.load(json_file)
    return json_dict

def dumpJSON(filename, map):
    with open(filename, "w+") as json_file:
        json.dump(map, json_file)
        
def generateSystemFileNameString(sys_config):
    compute_str = "{}tflops_".format(sys_config["matrix"]["float16"]["tflops"])
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