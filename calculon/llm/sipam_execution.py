"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  https://www.apache.org/licenses/LICENSE-2.0
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

import sys, copy, math, pprint
import matplotlib.pyplot as plt
import psutil
import datetime
import logging
import numpy as np
import multiprocessing as mp

import calculon
from calculon.llm.llm import Llm
from calculon.system import System
from calculon.util import pick
from calculon.llm import *
from scripts import utilities

def color(s:str):
  return f"\033[31m{s}\033[0m"

def dots(num:int):
  return "."*num

class SiPAMExecution(calculon.CommandLine):
  NAME = 'llm-sipam-execution'
  ALIASES = ['lse']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      SiPAMExecution.NAME, aliases=SiPAMExecution.ALIASES,
      help='run a search to find the optimal sipam execution')
    sp.set_defaults(func=SiPAMExecution.run_command)
    sp.add_argument('config_file', type=str,
                    help='File path to configuration')

  @staticmethod
  def run_command(logger, args):
    """
      Input: 
        app: original app
        syst: flops for a single CU, memory bw + capacity of a single MU
      Output:
        update syst and exe at every step of the optimization
    """
    ## Initial setup
    config = calculon.io.read_json_file(args.config_file)
    app = Llm.Application(calculon.io.read_json_file(config["model"]))
    syst = System(config["system"])
  
    worktype = config['worktype']
    datatype = config['datatype']
    max_batch_size = config['max_batch_size']
    max_num_procs = config['max_num_procs']
    exe_json = SiPAMExecution.get_init_exe(batch_size=max_batch_size, datatype=datatype, worktype=worktype)
    
    iteration = 0
    num_procs_list = []
    best_output = None
    while iteration < config['num_iter']:
      # Compile the model
      exe = Llm.Execution.from_json(exe_json)
      model = Llm(app, logger)
      model.compile(syst, exe)
      model.run_optim(syst)
      
      # Runs SiPAM optimization and update system params
      est_num_procs, config = SiPAMExecution.optimize(model, config)
      SiPAMExecution.set_syst_params(syst, config)
      
      # Build the parallel search params and find minimum num processors to fit the model
      num_procs, output = SiPAMExecution.find_min_num_procs(est_num_procs, max_num_procs, app, syst, max_batch_size, worktype, datatype)
      
      # Increment loop count and update execution params
      iteration += 1
      exe_json = output[0]['execution']
      stats = output[0]['stats']
      print(f"{dots(6)} {color(f'Iteration: {iteration}')}")
      print(f"{dots(6)} {color('Time')}: {stats['total_time_aggregate']}s")
      print(f"{dots(6)} {color(f'{num_procs} GPUs')} = {exe_json['tensor_par']}TP x {exe_json['pipeline_par']}PP x {exe_json['data_par']}DP")
      print(f"{dots(6)} {color('AI')}: {stats['arithmetic_intensity']['total']}, {color('Memory BW')}: {config['system']['mem1']['GBps']}GBps")
      print(f"{dots(6)} {color('Mem Needed')}: {stats['proc_mem_tier1_cap_req']/(1024**3)}GB, {color('Memory Cap')}: {config['system']['mem1']['GiB']}GB\n")
      
      best_output = output if not best_output or output[0]['stats']['total_time_aggregate'] < best_output[0]['stats']['total_time_aggregate'] else best_output 
      # Break if curr_num_procs is already in list
      if num_procs_list and num_procs in num_procs_list: break
      num_procs_list.append(num_procs)
      
    # write results to output file
    model_str = config["model"].split("/")[-1].split(".")[0]
    arch_str = utilities.generate_arch_file_name_string(output[0]['execution'])
    sys_str = utilities.generate_system_file_name_string(config["system"])
    output_dir = utilities.create_output_directory(config["output_file_dir"], model_str, arch_str)
    output_file_name = output_dir + sys_str + ".json"
    logger.info(f'[SiPAM] Output: {output_file_name}')
    calculon.io.write_json_file(output[0]['stats'], output_file_name)
    return 0
  
  @staticmethod
  def find_min_num_procs(est_num_procs, max_num_procs, app, syst, max_batch_size, worktype, datatype):
    print(f"[SiPAM] Searching for minimum number of processors ...")
    output = []
    while not output:
      print(f"[SiPAM] Current processor number = {est_num_procs}")
      params = SiPAMExecution.build_params(est_num_procs, app, syst, max_batch_size, worktype, datatype)
      output = SiPAMExecution.check_capacity(params)
      if not output: est_num_procs = int(1 << est_num_procs.bit_length())
    print(f"[SiPAM] Minimum number of processors = {output[0]['execution']['num_procs']}")
    return est_num_procs, output
    
  @staticmethod
  def find_min_num_procs_archive(est_num_procs, max_num_procs, app, syst, max_batch_size, worktype, datatype):
    ## Binary search to find minimum number of GPUs to fit the model
    power_min = int(math.log2(est_num_procs))
    power_max = int(math.log2(max_num_procs))
    optim = None
    output = []
    while power_min <= power_max:
      mid_power = (power_min + power_max) // 2
      n_curr = 2 ** mid_power
      params = SiPAMExecution.build_params(n_curr, app, syst, max_batch_size, worktype, datatype)
      output = SiPAMExecution.check_capacity(params)
      if output:
        optim = n_curr  # it's feasible — try to find smaller
        power_max = mid_power - 1
      else:
        power_min = mid_power + 1  # not enough GPUs — try more
    assert(optim != None), "[Error] No solutions found for num_procs."
    return optim, output

  @staticmethod
  def check_capacity(params):
    # Runs parallel searches
    start_time = datetime.datetime.now()
    with mp.Pool(psutil.cpu_count(logical=False)) as pool:
      searches = pool.starmap(SiPAMExecution.search, params)
    end_time = datetime.datetime.now()

    # Combines parallel search result into one data structure
    output = SiPAMExecution.process_results(searches)
    return output
  
  @staticmethod
  def get_batch_size(data_par, max_batch_size):
    """
    Returns the largest multiple of `data_par` that does not exceed `max_batch_size`.
    Returns None if `data_par` is larger than `max_batch_size`.
    """
    if data_par > max_batch_size:
      return None
    return (max_batch_size // data_par) * data_par

  @staticmethod
  def build_params(num_procs:int, app:Llm.Application, syst:System, 
                   max_batch_size: float, worktype: str, datatype: str):
    params = []
    for tp in Llm.get_all_tensor_parallelisms(num_procs, app.hidden, app.attn_heads):
      for pp in Llm.get_all_pipeline_parallelisms(num_procs, tp, app.num_blocks):
        dp = Llm.get_data_parallelism(num_procs, tp, pp)  
        for ppint in Llm.get_valid_pipeline_interleavings(app.num_blocks, pp):
          batch_size = SiPAMExecution.get_batch_size(dp, max_batch_size)
          if batch_size is None: continue
          for activation_recompute in pick(worktype == "training", ['full'], ['none']):
            for optimizer_sharding in [False]:
              for tensor_par_comm_type in ['rs_ag']:
                params.append(
                  (False, 1, False, num_procs,
                  datatype, worktype=="training", 
                  app, syst, tp, pp, dp,
                  ppint, batch_size, activation_recompute, 
                  optimizer_sharding, tensor_par_comm_type, 
                  [True], True, not True, not True))
    return params

  @staticmethod
  def search(debug, top_n, layers, num_procs, datatype, training,
             app, syst, tp, pp, dp, ppint, batch_size, activation_recompute,
             optimizer_sharding, tensor_par_comm_type, fused_acts, mbs_break,
             allow_tp_overlap, allow_dp_overlap):
    num_nets = syst.num_networks

    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0

    has_mem2 = False # syst.mem2.capacity > 0

    can_redo = Llm.can_redo_ag(tensor_par_comm_type, activation_recompute)
    for seq_par_ag_redo in pick(can_redo, [True, False], [False]):
      for data_par_overlap in pick(dp>1 and allow_dp_overlap, [True, False], [False]):
        for tensor_par_overlap in pick(tp>1 and allow_tp_overlap, ['none', 'ring', 'pipe'], ['none']):
          for weight_offload in pick(has_mem2, [True, False], [False]):
            if activation_recompute == 'full' or not has_mem2:
              activations_offloads = [False]
            else:
              activations_offloads = [True, False]
            for activations_offload in activations_offloads:
              for optimizer_offload in pick(has_mem2, [True, False], [False]):
                for fused_act in fused_acts:
                  for microbatch_size in Llm.get_valid_microbatch_sizes(
                      app.seq_size, tp, dp, batch_size, pp):
                    mbs_break_good = good_exe_count
                    for tn in pick(tp>1, range(num_nets), [0]):
                      for pn in pick(pp>1, range(num_nets), [0]):
                        for dn in pick(dp>1, range(num_nets), [0]):
                          exe_count += 1
                          exe_json = {
                            'num_procs': num_procs,
                            'tensor_par': tp,
                            'pipeline_par': pp,
                            'data_par': dp,
                            'tensor_par_net': tn,
                            'pipeline_par_net': pn,
                            'data_par_net': dn,
                            'batch_size': batch_size,
                            'microbatch_size': microbatch_size,
                            'datatype': datatype,
                            'fused_activation': fused_act,
                            'attention_type': 'multihead',
                            'activation_recompute': activation_recompute,
                            'pipeline_interleaving': ppint,
                            'optimizer_sharding': optimizer_sharding,
                            'tensor_par_comm_type': tensor_par_comm_type,
                            'tensor_par_overlap': tensor_par_overlap,
                            'seq_par_ag_redo': seq_par_ag_redo,
                            'data_par_overlap': data_par_overlap,
                            'weight_offload': weight_offload,
                            'activations_offload': activations_offload,
                            'optimizer_offload': optimizer_offload,
                            'training': training
                          }

                          if not debug:
                            try:
                              logger = logging.Logger('sub')
                              model = Llm(app, logger)
                              model.compile(syst, Llm.Execution.from_json(exe_json))
                              model.run_optim(syst)
                              stats = model.get_stats_json(layers)
                              good_exe_count += 1
                              curr = (stats['sample_rate_aggregate'], exe_json, stats)
                              best = SiPAMExecution.update_list(best, curr, top_n)
                            except Llm.Error as ex:
                              logger = logging.getLogger()
                              logger.debug(f'JSON:{exe_json}\nERROR:{ex}\n')
                              bad_exe_count += 1
                    if mbs_break and good_exe_count == mbs_break_good:
                      break
    return (best, exe_count, good_exe_count, bad_exe_count, tp, pp, dp)

  @staticmethod
  def process_results(searches):
    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0
    for cbest, ec, gec, bec, tp, pp, dp in searches:
      best = SiPAMExecution.update_list(best, cbest, 1)
      exe_count += ec
      good_exe_count += gec
      bad_exe_count += bec
      
    output = {}
    for index, run in enumerate(best):
      _, execution, stats = run
      output[index] = {
        'execution': execution,
        'stats': stats
      }
    return output

  @staticmethod
  def update_list(current, candidate, quantity):
    if not isinstance(candidate, list):
      current.append(candidate)
    else:
      current.extend(candidate)
    current.sort(reverse=True, key=lambda x: x[0]) # sort based on decreasing sample rate
    return current[:quantity]
  
  @staticmethod
  def get_init_exe(batch_size, datatype, worktype):
    exe_json = {
                'num_procs': 1,
                'tensor_par': 1,
                'pipeline_par': 1,
                'data_par': 1,
                'tensor_par_net': 0,
                'pipeline_par_net': 1,
                'data_par_net': 1,
                'batch_size': batch_size,
                'microbatch_size': 1,
                'datatype': datatype,
                'fused_activation': True,
                'attention_type': 'multihead',
                'activation_recompute': "full",
                'pipeline_interleaving': 1,
                'optimizer_sharding': False,
                'tensor_par_comm_type': "rs_ag",
                'tensor_par_overlap': "none",
                'seq_par_ag_redo': False,
                'data_par_overlap': False,
                'weight_offload': False,
                'activations_offload': False,
                'optimizer_offload': False,
                'training': worktype
              }
    return exe_json

  @staticmethod
  def optimize(model: Llm, curr_config: dict):
    optim_config = copy.deepcopy(curr_config)
    datatype = curr_config["datatype"]
    flops_matrix = curr_config["system"]["matrix"][datatype]["tflops"] * 1e12
    ai_list = model.get_arithmetic_intensity()
    ai = ai_list['total'] # matrix, vector, total, mean, median
    
    req_mem_bw_per_gpu_GBps = flops_matrix / ai / 1e9
    num_req_mu_per_gpu = int(np.ceil(req_mem_bw_per_gpu_GBps / curr_config["system"]["mem1"]["GBps_orig"]))
    per_gpu_mem_bw_GBps = num_req_mu_per_gpu * curr_config["system"]["mem1"]["GBps_orig"]
    per_gpu_mem_cap_GB = num_req_mu_per_gpu * curr_config["system"]["mem1"]["GiB_orig"]
    num_procs = int(np.ceil((model.get_mem_tier1_cap_req() + model.get_mem_tier2_cap_req()) / (1024**3) / per_gpu_mem_cap_GB))
    num_procs = 1<<(num_procs-1).bit_length() # nearest power of 2
    # num_procs = (num_procs + 1) // 2 * 2 * 10 # nearest multiple of 2

    min_num_mem_pic_per_gpu = 1
    max_num_mem_pic_per_gpu = optim_config["system"]["max_num_mem_pic_per_gpu"]
    per_pic_bw_GBps = optim_config["system"]["per_pic_bw_GBps"]
    per_pic_length_mm = optim_config["system"]["per_pic_length_mm"]
    total_length_mm = optim_config["system"]["total_length_mm"]
    num_mem_pic_per_gpu = max(min_num_mem_pic_per_gpu,
                            min(max_num_mem_pic_per_gpu,
                              int(np.ceil(per_gpu_mem_bw_GBps / per_pic_bw_GBps))))
    num_net_pic_per_gpu = (total_length_mm - (per_pic_length_mm * num_mem_pic_per_gpu)) // per_pic_length_mm # round down
    net_bw_GBps = num_net_pic_per_gpu * per_pic_bw_GBps
    
    optim_config["system"]["mem1"]["GiB"] = per_gpu_mem_cap_GB
    optim_config["system"]["mem1"]["GBps"] = per_gpu_mem_bw_GBps
    optim_config["system"]["mem2"]["GiB"] = 0
    optim_config["system"]["mem2"]["GBps"] = 0
    optim_config["system"]["mem2"]["ns"] = 0
    optim_config["system"]["networks"][0]["bandwidth"] = net_bw_GBps
    optim_config["system"]["networks"][1]["bandwidth"] = net_bw_GBps
    return num_procs, optim_config
    
  @staticmethod
  def set_syst_params(syst: System, config):
    syst.set_mem1_bandwidth(config["system"]["mem1"]["GBps"])
    syst.set_mem1_capacity(config["system"]["mem1"]["GiB"])
    # make sure mem2 is turned off
    syst.set_mem2_bandwidth(0) 
    syst.set_mem2_capacity(0)
    syst.set_net_bandwidth(config["system"]["networks"][0]["bandwidth"])

calculon.CommandLine.register(SiPAMExecution)
