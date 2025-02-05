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

import sys, os
import psutil
import datetime
import logging
import numpy as np
import multiprocessing as mp

import calculon
from calculon.llm.llm import Llm
from calculon.util import pick
from calculon.llm import *

class SiPAMExecution(calculon.CommandLine):
  NAME = 'llm-sipam-execution'
  ALIASES = ['lse']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      SiPAMExecution.NAME, aliases=SiPAMExecution.ALIASES,
      help='run a search to find the optimal sipam execution')
    sp.set_defaults(func=SiPAMExecution.run_command)
    sp.add_argument('max_batch_size', type=int,
                    help='Maximum batch size, will be largest multiple of DP')
    sp.add_argument('datatype', type=str, choices=System.supported_datatypes(),
                    help='The datatype to use')
    sp.add_argument('optim_iter', type=int, default=5,
                    help='Number of iterations to run for optimization')
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('output', type=str,
                    help='File path to the output file'
                    " ('*.csv', '*.csv.gz', '*.json', '*.json.gz')")
    sp.add_argument('-t', '--training', action='store_true',
                    help='Run optimization on training or inference')

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
    app = Llm.Application(calculon.io.read_json_file(args.application))
    syst = System(calculon.io.read_json_file(args.system))
    exe_json = SiPAMExecution.get_init_exe(batch_size=3072, microbatch_size=4, datatype=args.datatype, worktype="training")

    iteration = 0
    prev_num_procs = -1
    prev_ai_total = -1
    
    # store prev 10 results and check if curr AI is in one of them
    while iteration < args.optim_iter:
      exe = Llm.Execution.from_json(exe_json)
      model = Llm(app, logger)
      model.compile(syst, exe)
      model.run(syst)
      
      flops_matrix = syst.get_matrix_flops(args.datatype)
      flops_vector = syst.get_vector_flops(args.datatype)
      ai = model.get_arithmetic_intensity()
      ai_matrix = ai['matrix']
      # ai_vector = ai['vector']
      ai_total = ai['total']
      
      # req_mem_bw_per_gpu_GBps = max(flops_matrix / ai_matrix, flops_vector / ai_vector) / 1e9
      # req_mem_bw_per_gpu_GBps = min(flops_matrix / ai_matrix, flops_vector / ai_vector) / 1e9
      req_mem_bw_per_gpu_GBps = ai_total / ai_matrix / 1e9
      

      num_req_mu_per_gpu = int(np.ceil(req_mem_bw_per_gpu_GBps / (syst.get_mem1_bandwidth() / 1e9)))
      per_gpu_mem_bw_GBps = num_req_mu_per_gpu * syst.get_mem1_bandwidth() / 1e9
      per_gpu_mem_cap_GB = num_req_mu_per_gpu * syst.get_mem1_capacity() / (1024**3)

      syst.set_mem1_bandwidth(per_gpu_mem_bw_GBps)
      syst.set_mem1_capacity(per_gpu_mem_cap_GB)
      
      num_procs = int(np.ceil(model.get_total_req_mem_cap() / (1024**3) / per_gpu_mem_cap_GB))
      # num_procs = 1<<(num_procs-1).bit_length() # nearest power of 2
      num_procs = (num_procs + 1) // 2 * 2 # nearest multiple of 2

      print(iteration, exe_json["tensor_par"], exe_json["pipeline_par"], exe_json["data_par"],
            ai_matrix, model.get_total_req_mem_cap()/(1024**3), per_gpu_mem_bw_GBps, num_procs)
      print("\n")

      if ai_total == prev_ai_total: break
      else: prev_ai_total = ai_total
      
      params = []
      for tp in Llm.get_all_tensor_parallelisms(
          num_procs, app.hidden, app.attn_heads):
        for pp in Llm.get_all_pipeline_parallelisms(
            num_procs, tp, app.num_blocks):
          dp = Llm.get_data_parallelism(num_procs, tp, pp)
          for ppint in Llm.get_valid_pipeline_interleavings(app.num_blocks, pp):
            batch_size = SiPAMExecution.get_batch_size(dp, args.max_batch_size)
            if batch_size is None: continue
            for activation_recompute in pick(args.training, ['full'], ['none']):
              for optimizer_sharding in [False]:
                for tensor_par_comm_type in ['rs_ag']:
                  params.append(
                    (False, 1, False, num_procs,
                    args.max_batch_size, args.datatype, args.training, app, syst, tp, pp, dp,
                    ppint, batch_size, activation_recompute, optimizer_sharding,
                    tensor_par_comm_type, [True], True,
                    not True, not True))
                  
      # Runs parallel searches
      start_time = datetime.datetime.now()
      with mp.Pool(psutil.cpu_count(logical=False)) as pool:
        searches = pool.starmap(SiPAMExecution.search, params)
      end_time = datetime.datetime.now()

      # Combines parallel search result into one data structure
      best = []
      exe_count = 0
      good_exe_count = 0
      bad_exe_count = 0
      for cbest, ec, gec, bec, tp, pp in searches:
        best = SiPAMExecution.update_list(best, cbest, 1)
        exe_count += ec
        good_exe_count += gec
        bad_exe_count += bec

      logger.info(f'Total executions: {exe_count}')
      logger.info(f'Good executions: {good_exe_count}')
      logger.info(f'Bad executions: {bad_exe_count}')
      calc_rate = exe_count / (end_time - start_time).total_seconds()
      logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')

      output = {}
      for index, run in enumerate(best):
        _, execution, stats = run
        output[index] = {
          'execution': execution,
          'stats': stats
        }
        
      iteration += 1
      exe_json = output[0]['execution']
    
    sys.exit()
    ## last step
    if calculon.io.is_json_extension(args.output):
      logger.info(f'Output: {args.output}')
      calculon.io.write_json_file(output, args.output)
    else:
      assert False, f'Unknown file type: {args.output}'

    return 0
  
  
  @staticmethod
  def get_batch_size(data_par, max_batch_size):
    if data_par > max_batch_size:
      return None
    last = data_par
    while True:
      if last + data_par > max_batch_size:
        return last
      else:
        last += data_par

  @staticmethod
  def search(debug, top_n, layers, num_procs, max_batch_size, datatype, training,
             app, syst, tp, pp, dp, ppint, batch_size, activation_recompute,
             optimizer_sharding, tensor_par_comm_type, fused_acts, mbs_break,
             allow_tp_overlap, allow_dp_overlap):
    num_nets = syst.num_networks

    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0

    has_mem2 = False # syst.mem2.capacity > 0

    can_redo = Llm.can_redo_ag(tensor_par_comm_type,
                               activation_recompute)
    for seq_par_ag_redo in pick(can_redo, [True, False], [False]):
      for data_par_overlap in pick(dp>1 and allow_dp_overlap, [True, False],
                                   [False]):
        for tensor_par_overlap in pick(tp>1 and allow_tp_overlap,
                                       ['none', 'ring', 'pipe'], ['none']):
          for weight_offload in pick(has_mem2, [True, False], [False]):
            if activation_recompute == 'full' or not has_mem2:
              activations_offloads = [False]
            else:
              activations_offloads = [True, False]
            for activations_offload in activations_offloads:
              for optimizer_offload in pick(has_mem2, [True, False],
                                            [False]):
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
                              model.compile(
                                syst,
                                Llm.Execution.from_json(exe_json))
                              model.run(syst)
                              stats = model.get_stats_json(layers)
                              good_exe_count += 1
                              curr = (stats['sample_rate'], exe_json, stats)
                              best = SiPAMExecution.update_list(best, curr, top_n)
                            except Llm.Error as ex:
                              logger = logging.getLogger()
                              logger.debug(f'JSON:{exe_json}\nERROR:{ex}\n')
                              bad_exe_count += 1
                    if mbs_break and good_exe_count == mbs_break_good:
                      break
    return (best, exe_count, good_exe_count, bad_exe_count, tp, pp)

  @staticmethod
  def update_list(current, candidate, quantity):
    if not isinstance(candidate, list):
      current.append(candidate)
    else:
      current.extend(candidate)
    current.sort(reverse=True, key=lambda x: x[0])
    return current[:quantity]
  
  @staticmethod
  def get_init_exe(batch_size, microbatch_size, datatype, worktype):
    exe_json = {
                'num_procs': 1,
                'tensor_par': 1,
                'pipeline_par': 1,
                'data_par': 1,
                'tensor_par_net': 0,
                'pipeline_par_net': 1,
                'data_par_net': 1,
                'batch_size': batch_size,
                'microbatch_size': microbatch_size,
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
  def optimize(model, syst):
    flops_matrix = syst.get_matrix_flops()
    flops_vector = syst.get_vector_flops()
    ai = model.get_arithmetic_intensity()
    ai_matrix = ai['matrix']
    ai_vector = ai['vector']

    req_mem_bw_per_gpu_GBps = max(flops_matrix / ai_matrix, 
                                  flops_vector / ai_vector)

    num_req_mu_per_gpu = int(np.ceil(req_mem_bw_per_gpu_GBps / syst.get_mem1_bandwidth()))
    

calculon.CommandLine.register(SiPAMExecution)
