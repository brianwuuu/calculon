import os, sys, stat, json, itertools

def nearest_pow_of_2(x):
    return 1<<(x-1).bit_length()

# Parse JSON file into dictionary object
def parseJSON(filename):
    with open(filename) as json_file: 
        json_dict = json.load(json_file)
    return json_dict

def dumpJSON(filename, map):
    with open(filename, "w+") as json_file:
        json.dump(map, json_file, indent=2)
        
def createDirectory(directory_path):
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

def createOutputDirectory(output_dir, model_dir, arch_dir):
    createDirectory(output_dir + model_dir)
    createDirectory(output_dir + model_dir + "/" + arch_dir)
    createDirectory(output_dir + model_dir + "/" + arch_dir + "/")
    return output_dir + model_dir + "/" + arch_dir + "/" 

def writeStringToFile(filename, string):
    with open(filename, "w+") as f:
        f.write(string)
        
def cartesianProduct(param_list):
    return [x for x in itertools.product(*param_list)]
        
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
        
# Generate the bash script used to run simulations in Netbench
def generateBashScript(exec_dir, config_file_list, exp_name=""):
    # Construct the string builder
    str_builder = "cd $CALCULON_HOME\n\n"
    str_builder += "export PYTHONPATH=.\n\n"
    exec_prefix = "./bin/calculon llm"
    exec_file_name = "automated_execution.sh"
    # Write the script to the .sh file
    for config_file in config_file_list:
        assert(len(config_file) == 3), "[Error] Must have 3 file inputs, now have {}".format(len(config_file))
        str_builder += (exec_prefix + " " + config_file[0] + " " + config_file[1] + " " + config_file[2] + "\n")
    with open(exec_dir + exec_file_name, "w+") as f:
        f.write(str_builder)
    print("[Setup] Generate bash script to {}".format(exec_dir + exec_file_name))
    st = os.stat(exec_dir + exec_file_name)
    os.chmod(exec_dir + exec_file_name, st.st_mode | stat.S_IEXEC)
    return 

def generateExecutionScript(exec_dir, exp_name="", bash_script_names=[]):
    if not bash_script_names: return
    file_name = exp_name + "_nohup" +".sh"
    str_builder = ""
    for name in bash_script_names:
        str_builder += "nohup .{} > ../logs{} 2>&1 &\n".format(name, name)
    with open(exec_dir + "/" + file_name, "w+") as f:
        f.write(str_builder)
    print("[Setup] Generate nohup script to {}".format(exec_dir+"/"+file_name))
    st = os.stat(exec_dir + "/" + file_name)
    os.chmod(exec_dir + "/" + file_name, st.st_mode | stat.S_IEXEC)