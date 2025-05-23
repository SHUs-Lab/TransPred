"""
Benchmark Utilities

Modified based on the reference from:
https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa

"""
import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple

from alpa import get_global_cluster, set_global_virtual_physical_mesh
from alpa.model.moe import FlaxMoEForLMModule, MoEConfig, TrainState
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.util import print_used_time
import optax

from benchmark_gpt import get_train_step
from util import compute_moe_parameter_count, compute_moe_tflops
from benchmark_parallel_utils import (
    get_pipeshard_parallel_method,
    compile_and_benchmark_pipeshard_training_executable)

MoEModelConfig = namedtuple("MoEModelConfig", [
    "seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size",
    "num_experts", "expert_group_size"
])

moe_specs = {
    #                      S,    H,   L, head, V,   E,  S_
    "380M": MoEModelConfig(1024, 768, 8, 16, 32000, 8, 2048),
    "690M": MoEModelConfig(1024, 768, 8, 16, 32000, 16, 2048),
    "1.3B": MoEModelConfig(1024, 768, 16, 16, 32000, 16, 2048),
    "2.4B": MoEModelConfig(1024, 1024, 16, 16, 32000, 16, 2048),
    "2.6B": MoEModelConfig(1024, 768, 32, 16, 32000, 16, 2048),
}

def create_train_state(rngkey, model, dtype, batch):
    params = model.init_dummy(rngkey, batch["input_ids"],
                              batch["attention_mask"], batch["token_type_ids"],
                              batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.adafactor(learning_rate=1e-2,
                         weight_decay_mask=weight_decay_mask)

    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              use_master_copy=(dtype == jnp.float16),
                              dynamic_scale=None)
    return state


def prepare_moe_input_and_model(benchmark_case,
                                add_manual_remat=None,
                                add_manual_layer_marker=None,
                                num_manual_pipeline_stages=None,
                                correct_expert_group_size=True):
    print_used_time(None)
    print(benchmark_case)
    (batch_size, model_config, num_micro_batches,_,_) = benchmark_case
    (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts,
     expert_group_size) = model_config
    dtype = jnp.float16
    tie_word_embeddings = False

    if correct_expert_group_size:
        rang_factor = 1
        expected_expert_group_size = min(
            expert_group_size,
            batch_size * seq_len // num_micro_batches // 1 // rang_factor)
        if expected_expert_group_size != expert_group_size:
            print("- Expected expert group size should be {}, "
                  "but got {}. Will reset it".format(expected_expert_group_size,
                                                     expert_group_size))
            expert_group_size = expected_expert_group_size

    # Prepare input batch
    batch = {
        "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    model = FlaxMoEForLMModule(
        MoEConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 8,  # this is specific to gspmd.
            num_attention_heads=num_heads,
            max_position_embeddings=seq_len,
            vocab_size=vocab_size,
            expert_group_size=expert_group_size,
            expert_number=num_experts,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=add_manual_remat,
            add_manual_pipeline_markers=add_manual_layer_marker,
            pipeline_mp_size=num_manual_pipeline_stages,
        ),
        dtype=dtype)

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, dtype, batch)
    print_used_time("Create train state")
    return state, batch, rngkey


def compute_moe_statistics(benchmark_case, latencies, num_devices):
    batch_size = benchmark_case.batch_size
    (seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts,
     expert_group_size) = benchmark_case.model_config
    use_remat = benchmark_case.parallel_args.use_remat

    tflops = compute_moe_tflops(batch_size,
                                seq_len,
                                num_layers,
                                hidden_size,
                                expert_group_size,
                                vocab_size,
                                num_experts,
                                num_devices,
                                np.mean(latencies),
                                checkpoint_activations=use_remat)
    parameter_count = compute_moe_parameter_count(num_layers,
                                                  hidden_size,
                                                  vocab_size,
                                                  num_experts,
                                                  mlp_factor=8)
    return tflops, parameter_count


def benchmark_moe_3d_internal(benchmark_case,
                              niter,
                              num_hosts,
                              num_devices_per_host,
                              profile_driver_time=False):
    predict_mode = benchmark_case.parallel_mode=='predict'
    # Connect to the cluster
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh)

    # Parallel configs
    (method, add_manual_remat, add_manual_layer_marker,
     num_manual_pipeline_stages) = get_pipeshard_parallel_method(
         benchmark_case,
         allow_mixed_mesh_shape=True)

    state, batch, rngkey = prepare_moe_input_and_model(
        benchmark_case,
        add_manual_remat=add_manual_remat,
        add_manual_layer_marker=add_manual_layer_marker,
        num_manual_pipeline_stages=num_manual_pipeline_stages)

    train_step = get_train_step(method)

    (latencies, max_mem_allocated, compilation_times,
     executable) = compile_and_benchmark_pipeshard_training_executable(
         niter,
         train_step,
         state, (batch, rngkey),
         profile_driver_time=profile_driver_time, predict=predict_mode
    )
     
    if predict_mode:
        print('DONE', latencies)


    tflops, parameter_count = compute_moe_statistics(benchmark_case, latencies,
                                                     virtual_mesh.num_devices)

    (compute_cost_file_name, forward_stage_layer_ids, submesh_shapes,
     logical_mesh_shapes, autosharding_option_dicts) = get_last_dp_result()
    metadata = {
        "compilation_times": compilation_times,
        "compute_cost_file_name": compute_cost_file_name,
        "forward_stage_layer_ids": forward_stage_layer_ids,
        "submesh_shapes": submesh_shapes,
        "logical_mesh_shapes": logical_mesh_shapes,
        "autosharding_option_dicts": autosharding_option_dicts,
    }

    return parameter_count, max_mem_allocated, latencies, tflops, metadata
