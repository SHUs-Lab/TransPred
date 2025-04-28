import time
import argparse
import numpy as np
from alpa.util import (to_str_round, GB)

from benchmark_parallel_utils import (ParallelArgs, BenchmarkCase)
from benchmark import benchmark_one_case, auto_stage_option
from benchmark_moe import moe_specs
from benchmark_gpt import gpt_specs

moe_model = 'moe'
gpt_model = 'gpt'
benchmark_models = [moe_model, gpt_model]

def benchmark(args):
    model_type = args.model
    num_hosts = 2
    num_devices_per_host = 2
    
    num_micro_batches = 64
    num_auto_layers = 32
    prefer_reduce_scatter = True
    use_remat = True
    num_iteration = 3
    
    num_gpus = num_hosts * num_devices_per_host
    max_global_batch_size = 1024
    
    if args.profile_amt and args.profile_amt != "":
        auto_stage_option['profile_amount'] = True
        auto_stage_option['profile_amount'] = args.profile_amt
    
    # run benchmark with historical data from different platform
    if args.history_path and args.history_path != "":
        auto_stage_option['use_history_data'] = True
        auto_stage_option['historical_data_path'] = args.history_path
    
    if args.error_lim and args.error_lim != "":
        auto_stage_option['fs_cutoff'] = args.error_lim
        
    if model_type == moe_model:
        model_config = moe_specs['2.6B']
        auto_stage_option['compile_limit'] = 150
        # auto_stage_option['full_data_file'] = 'history_data/moe_l32_m64_data_all.pkl'
        # auto_stage_option['full_data_file'] = 'history_data/moe_data_all.pkl'
        
    elif model_type == gpt_model:
        model_config = gpt_specs['1.3B']
        auto_stage_option['compile_limit'] = 250000
        # auto_stage_option['full_data_file'] = 'history_data/gpt_lat_data_64_32_1.3B.pkl'
        # auto_stage_option['full_data_file'] = 'history_data/gpt_data_all.pkl'
    
    else:
        raise Exception("invalid model type")

    print(auto_stage_option)
    
    
    parallel_args = ParallelArgs(prefer_reduce_scatter, use_remat,
                               num_auto_layers, auto_stage_option)

    benchmark_case = BenchmarkCase(
        max_global_batch_size,
        model_config,
        num_micro_batches,
        'search',
        parallel_args
    )

    # Run one case
    print("Working on case: {}".format(str(benchmark_case)))
    result = benchmark_one_case(
        model_type,
        benchmark_case,
        num_iteration,
        num_hosts,
        num_devices_per_host)

    (parameter_count, peak_mem, latencies, tflops, metadata) = result
    
    print(result)
    print('Latency: ', f"{np.mean(latencies):.3f}")
    
    values = [
        model_type, model_config, num_micro_batches, num_gpus,
        parallel_args, f"{np.mean(latencies):.3f}",
        f"{np.std(latencies):.3f}", f"{parameter_count/1e9:.3f}B",
        f"{tflops:.2f}", f"{peak_mem/GB:.3f}",
        to_str_round(metadata, 2)
    ]
    
    print(values)

    time.sleep(0.1)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        choices=benchmark_models,
                        type=str,
                        required=True)

    
    parser.add_argument("--history-path",
                        type=str,
                        default="",
                        required=False)
    
    parser.add_argument("--error-lim",
                        type=int,
                        default=5,
                        required=False)
    
    parser.add_argument("--profile-amt",
                        type=float,
                        default=0.3,
                        required=False)
        
    args = parser.parse_args()
    benchmark(args)