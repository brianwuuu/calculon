import copy
import utilities
import itertools

BASE_DIRECTORY = "/Users/bwu/src/calculon/"
OUTPUT_DIRECTORY = BASE_DIRECTORY + "temp/"
SYSTEM_DIRECTORY = BASE_DIRECTORY + "systems/"
MODEL_DIRECTORY = BASE_DIRECTORY + "models/"
ARCH_DIRECTORY = BASE_DIRECTORY + "examples/"
EXECUTION_DIRECTORY = BASE_DIRECTORY + "execution/"

def generate_model_configs(model_params : dict, **kwargs):
    """
    Args:
        model_params: [(hidden, attn_size, num_blocks)]
    """
    seq_size = 2048 # 2048 (GPT-3), 8192
    model_config_files = []
    for param_dict in model_params:
        model = param_dict["model"]
        model_base_filename = MODEL_DIRECTORY + model + ".json"
        model_base = utilities.parseJSON(model_base_filename)
        new_model = copy.deepcopy(model_base)
        new_model["seq_size"] = seq_size 
        if 'hidden' in param_dict: new_model["hidden"] = param_dict['hidden']
        if 'attn_size' in param_dict: new_model["attn_size"] = param_dict['attn_size']
        if 'num_blocks' in param_dict: new_model["num_blocks"] = param_dict['num_blocks']
        if 'hidden' in param_dict: new_model["feedforward"] = 4 * param_dict['hidden']
        if 'num_blocks' in param_dict: new_model["attn_heads"] = param_dict['num_blocks']
        if not any(param in param_dict.keys() for param in ['hidden', 'attn_size', 'num_blocks']):
            model_config_file = model_base_filename
        else:
            model_filename = utilities.generateModelFileNameString(new_model) + ".json"
            model_config_file = MODEL_DIRECTORY + model_filename
            utilities.dumpJSON(model_config_file, new_model)
        model_config_files.append(model_config_file)
    return model_config_files

def generate_arch_configs(arch_params, **kwargs):
    """
    Args:
        arch_params: [(num_procs, tensor_par, pipe_par, data_par)]
    """
    arch = "4096_t8_p64_d8_mbs4_full" # "3072_t4_p64_d12_mbs4_full"
    arch_base_filename = ARCH_DIRECTORY + arch + ".json"
    arch_base = utilities.parseJSON(arch_base_filename)
    arch_config_files = []
    if not arch_params: arch_config_files.append(arch_base_filename)
    for num_proc, tensor_par, pipe_par, data_par in arch_params:
        new_arch = copy.deepcopy(arch_base)
        new_arch["num_procs"] = num_proc
        new_arch["tensor_par"] = tensor_par
        new_arch["pipeline_par"] = pipe_par
        new_arch["data_par"] = data_par
        new_arch["tensor_par_net"] = 0
        new_arch["pipeline_par_net"] = 1
        new_arch["data_par_net"] = 1
        # offloading
        new_arch["weight_offload"] = False
        new_arch["activations_offload"] = False
        new_arch["optimizer_offload"] = False
        # parallelization params
        new_arch["optimizer_sharding"] = True if data_par > 1 else False
        if "worktype" in kwargs: 
            new_arch["training"] = False if kwargs['worktype'] == "inference" else True
        new_arch["activation_recompute"] = 'none' if new_arch["training"] == False else "full"
        arch_filename = utilities.generateArchFileNameString(new_arch)
        arch_config_file = ARCH_DIRECTORY + arch_filename + ".json"
        utilities.dumpJSON(arch_config_file, new_arch)
        arch_config_files.append(arch_config_file)
    return arch_config_files

def generate_system_configs(gpu, mem_params, net_params):
    """
    Args:
        mem_params: mem1 GB, GBps, ns; mem2 GB, GBps, ns
        net_params: net1 GBps, latency, proc_util; net2 GBps, latency, proc_util
    """
    if gpu == "h100": gpu = "h100_80g_nvl8" 
    elif gpu == "a100": gpu = "a100_80g"
    elif gpu == "b100": gpu = "b100_80g"
    else: raise Exception(f"[Error] GPU {gpu} not known")
    system_base_filename = SYSTEM_DIRECTORY + gpu + ".json"
    system_base = utilities.parseJSON(system_base_filename)
    sys_config_files = []
    for mem_param, net_param in itertools.zip_longest(mem_params, net_params, fillvalue=None):
        new_system = copy.deepcopy(system_base)
        if "mem1_GB" in mem_param: new_system["mem1"]["GiB"] = mem_param['mem1_GB']
        if "mem1_GBps" in mem_param: new_system["mem1"]["GBps"] = mem_param['mem1_GBps']
        if "mem1_ns" in mem_param: new_system["mem1"]["ns"] = mem_param['mem1_ns']
        if "mem2_GB" in mem_param: new_system["mem2"]["GiB"] = mem_param['mem2_GB']
        if "mem2_GBps" in mem_param: new_system["mem2"]["GBps"] = mem_param['mem2_GBps']
        if "mem2_ns" in mem_param: new_system["mem2"]["ns"] = mem_param['mem2_ns']
        if "net1_GBps" in net_param: new_system["networks"][0]["bandwidth"] = net_param['net1_GBps']
        if "net1_ns" in net_param: new_system["networks"][0]["latency"] = net_param['net1_ns'] / 1e9
        if "net1_eff" in net_param: new_system["networks"][0]["efficiency"] = net_param['net1_eff']
        if "net2_GBps" in net_param: new_system["networks"][1]["bandwidth"] = net_param['net2_GBps']
        if "net2_ns" in net_param: new_system["networks"][1]["latency"] = net_param['net2_ns'] / 1e9
        if "net2_eff" in net_param: new_system["networks"][1]["efficiency"] = net_param['net2_eff']
        new_system["networks"][0]["size"] = 32768
        new_system["processing_mode"] = "no_overlap" # roofline, no_overlap
        system_filename = utilities.generateSystemFileNameString(new_system)
        sys_config_file = SYSTEM_DIRECTORY + system_filename
        utilities.dumpJSON(sys_config_file, new_system)
        sys_config_files.append(sys_config_file)
        # output_dir = utilities.createOutputDirectory(OUTPUT_DIRECTORY, model_filename, arch, gpu)
        # sys_config_files.append(SYSTEM_DIRECTORY + system_filename + " " + output_dir + system_filename)
    return sys_config_files

def generate_output_files(model_config_files, arch_config_files, sys_config_files):
    config_files = utilities.zipConfigs([model_config_files, arch_config_files, sys_config_files])
    new_config_files = []
    for model, arch, system in config_files:
        model_str = (model.split("/")[-1]).split(".")[0]
        arch_str = (arch.split("/")[-1]).split(".")[0]
        sys_str = system.split("/")[-1]
        output_dir = utilities.createOutputDirectory(OUTPUT_DIRECTORY, model_str, arch_str)
        output_str = system + " " + output_dir + sys_str
        new_config_files.append((model, arch, output_str))
    return new_config_files

def setup_experiment(mem_params, net_params, model_params, arch_params, **kwargs):
    gpu = kwargs["gpu"]
    exp_name = kwargs["exp_name"] if "exp_name" in kwargs else ""
    sys_config_files = generate_system_configs(gpu, mem_params, net_params)
    model_config_files = generate_model_configs(model_params)
    arch_config_files = generate_arch_configs(arch_params, **kwargs)
    config_files = generate_output_files(model_config_files, arch_config_files, sys_config_files)
    bash_script = utilities.generateBashScript(EXECUTION_DIRECTORY, config_files, exp_name=exp_name)
    # utilities.generateExecutionScript(EXECUTION_DIRECTORY, bash_script_names)

def get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params, **kwargs):
    model_config_files = generate_model_configs(model_params)
    arch_config_files = generate_arch_configs(arch_params, **kwargs)
    sys_config_files = generate_system_configs(gpu, mem_params, net_params)
    config_files = zip(model_config_files, arch_config_files, sys_config_files)
    return list(config_files)