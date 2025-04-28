"""
Functionalities about profiling the stages.

Modified based on the reference from:
https://github.com/alpa-projects/alpa/blob/main/alpa/pipeline_parallel/stage_profiling.py

"""
import os
from time import time
from datetime import datetime
import logging
import pickle
from typing import Dict, Sequence, Tuple
import numpy as np
import tqdm
import random
import itertools
import warnings
from pathlib import Path

from ray.exceptions import RayActorError

from jax import core
from jax.core import Var
import torch

from alpa.device_mesh import (VirtualPhysicalMesh, get_global_cluster)
from alpa.global_env import global_config
from alpa.pipeline_parallel.apply_grad import APPLY_GRAD_MARKER_SUFFIX
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, merge_marked_jaxprs_with_named_call, merge_unmarked_with_call)
from alpa.shard_parallel.auto_sharding import (AutoShardingOption,
                                               LogicalDeviceMesh)
from alpa.timer import timers
from alpa.util import (jaxpr_to_hlo, OrderedSet)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="networkx backend defined more than once: nx-loopback")
    import networkx as nx
from transpred.model.train import run, pred, create_model
from alpa.pipeline_parallel.stage_profiling import (
    ModuleProfileConfig,
    _get_layer_flops_prefix_sum,
    select_module_layers,
    ApplyGradConfig,
    CompileConfig,
    StageConfig,
    profile_all,
    compile_all,
    check_profile_results_consistent
)
from alpa.util import (retrieve_placement_group)
from alpa.mesh_profiling import (ProfilingResultDatabase,)
from alpa.pipeline_parallel.stage_profiling import (
    HloCostModelProfileWorkerPool,
    ProfileWorkerPool,
    ray,
    generate_module_profile_result,
    CompileOutput,
)

from scipy.optimize import curve_fit
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

last_compute_cost_file_name = None

INFINITY_N_STAGES = 2**20
GB = 1024**3

def get_training_stages_to_profile(layers, prob=1, reduce=0, exclude=[], all_st_ends=None):
    print("- Generate all stage infos (Jaxpr -> HLO)")
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    expected_stages_num = int(prob * (num_layers * (num_layers + 1) / 2))

    st_end = []

    min_wid_limit = 0 if prob==1 else 2 # stages should have minimum 2 layers
    print('exclude', exclude)
    max_wid_limit = num_layers - reduce

    wid_step = max(int(num_layers/expected_stages_num), 1)

    print(expected_stages_num, num_layers)
    print(min_wid_limit, max_wid_limit, wid_step)

    wid_range = list(range(min_wid_limit, max_wid_limit, wid_step))

    if all_st_ends != None:
        st_end = all_st_ends
    else:
        for wid in wid_range:
        # for wid in range(5):
            dds = list(range(num_layers - wid))

            count = int((num_layers - wid) * prob)
            count = max(1, count)
            starts = random.sample([el for el in dds if el not in exclude], count)
            # print(dds, wid)
            st_end += [(i, i + wid) for i in starts]
        # print([(i, i + wid) for i in starts])

        if len(st_end) > expected_stages_num:
            st_end = random.sample(st_end, expected_stages_num)
    return st_end

def process_training_stages_2d(st_end, layers,
                                accumulator_mapping,
                                acc_grad_invars,
                                acc_grad_outvars,
                                apply_grad_layers,
                                apply_grad_global_info,
                                mesh_id,
                                autosharding_configs,
                                mesh_num_devices,
                                cluster_size,
                                compile=True):
    print("- Generate all stage infos (Jaxpr -> HLO)")
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2

    indices = list(range(2 * num_layers))
    computation_source_ratio = mesh_num_devices / cluster_size
    is_full_mesh = computation_source_ratio == 1
    stages = []

    if is_full_mesh:
        st_end = [(0, num_layers-1)]

    len_stend = len(st_end)

    for elti in tqdm.tqdm(range(0, len_stend)):
        elt = st_end[elti]
        start = elt[0]
        end = elt[1]

        forward_layer_indices = indices[start:end + 1]
        backward_layer_indices = indices[2 * num_layers - end -
                                            1:2 * num_layers - start]
        selected_apply_grad_layers = [
            apply_grad_layers[idx]
            for idx in forward_layer_indices
            if apply_grad_layers[idx] is not None
        ]
        stage_name = f"stage_{start}_{end}"
        stage_config, mmjpr= generate_stage_info(
            layers, [forward_layer_indices, backward_layer_indices],
            accumulator_mapping, acc_grad_invars, acc_grad_outvars,
            stage_name, selected_apply_grad_layers, apply_grad_global_info, compile=compile)
        for config_idx, autosharding_config in enumerate(
                autosharding_configs):
            if autosharding_config is not None:
                stage_indices = (start, end, mesh_id, config_idx)
                stages.append(
                    (stage_indices, stage_config, autosharding_config, mmjpr))
    return stages

def generate_training_stages_2d(layers,
                                layer_flops_prefix_sum,
                                accumulator_mapping,
                                acc_grad_invars,
                                acc_grad_outvars,
                                apply_grad_layers,
                                apply_grad_global_info,
                                mesh_id,
                                autosharding_configs,
                                mesh_num_devices,
                                cluster_size,
                                stage_imbalance_tolerance=np.inf, compile=True, prob=1, reduce=0, exclude=[], all_st_ends=None):
    print("- Generate all stage infos (Jaxpr -> HLO)")
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    expected_stages_num = int(prob * (num_layers * (num_layers + 1) / 2))
    indices = list(range(2 * num_layers))
    computation_source_ratio = mesh_num_devices / cluster_size
    is_full_mesh = computation_source_ratio == 1
    tot_flops = layer_flops_prefix_sum[2 * num_layers]
    stages = []

    st_end = []
    end = num_layers

    min_wid_limit = 0 if prob==1 else 2 # stages should have minimum 2 layers
    
    max_wid_limit = num_layers - reduce

    wid_step = max(int(num_layers/expected_stages_num), 1)

    wid_range = list(range(min_wid_limit, max_wid_limit, wid_step))

    if all_st_ends != None:
        st_end = all_st_ends
    else:
        for wid in wid_range:
        # for wid in range(5):
            dds = list(range(num_layers - wid))

            count = int((num_layers - wid) * prob)
            count = max(1, count)
            starts = random.sample([el for el in dds if el not in exclude], count)
            # print(dds, wid)
            st_end += [(i, i + wid) for i in starts]

        if len(st_end) > expected_stages_num:
            st_end = random.sample(st_end, expected_stages_num)

    # DATAGEN: generate for full mesh
    if is_full_mesh:
        st_end = [(0, num_layers-1)]

    len_stend = len(st_end)

    for elti in tqdm.tqdm(range(0, len_stend)):
        elt = st_end[elti]
        start = elt[0]
        end = elt[1]

        forward_layer_indices = indices[start:end + 1]
        backward_layer_indices = indices[2 * num_layers - end -
                                            1:2 * num_layers - start]
        selected_apply_grad_layers = [
            apply_grad_layers[idx]
            for idx in forward_layer_indices
            if apply_grad_layers[idx] is not None
        ]
        stage_name = f"stage_{start}_{end}"
        stage_config, mmjpr= generate_stage_info(
            layers, [forward_layer_indices, backward_layer_indices],
            accumulator_mapping, acc_grad_invars, acc_grad_outvars,
            stage_name, selected_apply_grad_layers, apply_grad_global_info, compile=compile)
        for config_idx, autosharding_config in enumerate(
                autosharding_configs):
            if autosharding_config is not None:
                stage_indices = (start, end, mesh_id, config_idx)
                stages.append(
                    (stage_indices, stage_config, autosharding_config, mmjpr))
                    # (stage_indices, stage_config, autosharding_config))
    return stages


def _jaxpr_graph_g(jpl, G=None, id_names=None, jtype=0, md={'invars':{}, 'outvars':[]}, jpridx=0, pref = "", jaxpr_invars=[], remat=False, update_nodes={}):

    if jpl is None:
        return None, None

    if jtype == 0:
        jaxpr = core.Jaxpr(
            constvars=list(jpl.consts_dir.keys()),
            invars=jpl.invars,
            outvars=jpl.outvars,
            eqns=jpl.eqns,
        )
    elif jtype == 1:
        jaxpr = jpl
    else:
        raise Exception("invalid")

    def get_ivr(dict, key):
        st = str(key)
        while st in dict:
            st = dict[st]
        return st

    def get_ovr(dict, key):
        new_dict = {}
        for v in dict:
            k = dict[v]
            new_dict[k] = v

        return get_ivr(new_dict, key)

    def get_ovrs(dict, key):
        ovrs = []
        for k in dict:
            v = dict[k]
            if v == key:
                ovrs.append(k)

        if len(ovrs) == 0:
            return [key]
        else:
            return ovrs

    if G is None:
        G = nx.DiGraph()

    if id_names is None:
        id_names = (f'id{id}' for id in itertools.count())

    for v in jaxpr.constvars:
        G.add_node(
            pref + str(v),
            label=core.raise_to_shaped(v.aval).str_short(),
            shape=v.aval.shape,
            dtype=str(v.aval.dtype),
            type='const',
            remat=remat,
        )
    if jtype == 0:
        jaxpr_invars = [str(v) for v in jaxpr.invars]
        # TODO: FIX this is needed for combined layer stages generation
        for v in jaxpr.invars:
            G.add_node(
                str(v),
                label=str(v) + v.aval.str_short(),
                shape=v.aval.shape,
                dtype=str(v.aval.dtype),
                type='invar',
                remat=remat,
            )

    for eqn in jaxpr.eqns:
        if str(eqn.primitive) == 'pipeline_marker':
            for i, inv in enumerate(eqn.invars):

                v = eqn.outvars[i]

                if not G.has_node(str(v)):
                    edge_from = str(inv)
                    ntype = 'intermediate'

                    if jtype == 0 and eqn.params['mark_type'] == 'start' and str(inv) in jaxpr_invars:
                        edge_from = 'in ' + str(inv)
                        ntype = 'invar'

                    if eqn.params['mark_type'] == 'end' and str(inv) in update_nodes:
                        update_nodes[str(v)] = str(inv)
                        continue

                    G.add_node(
                        str(v),
                        label=str(v) + v.aval.str_short(),
                        shape=v.aval.shape,
                        dtype=str(v.aval.dtype),
                        type=ntype,
                        remat=remat,
                    )
                    if jtype == 0 and eqn.params['mark_type'] == 'start' and str(inv) in jaxpr_invars:
                        continue

                    G.add_edge(edge_from, str(v))
            continue

        if str(eqn.primitive) == 'custom_jvp_call':
            sub_jaxpr = eqn.params['call_jaxpr'].jaxpr
            outv, = eqn.outvars
            G.add_node(
                pref + str(outv),
                label=str(sub_jaxpr.eqns[0].primitive),
                shape=outv.aval.shape,
                dtype=str(outv.aval.dtype),
                type='op_node',
                remat=remat,
            )

            for v in [sub_jaxpr.eqns[0].invars[1]] + eqn.invars:
                if jtype != 0 and v in jaxpr.invars:
                    ivr = get_ivr(md['invars'], v)

                    if isinstance(v, core.Literal):
                        id_name = next(id_names)
                        G.add_node(
                            pref + id_name,
                            label=v,
                            shape=v.aval.shape,
                            dtype=str(v.aval.dtype),
                            type='literal',
                            remat=remat,
                        )
                        edge_from = pref + id_name
                    elif G.has_node(pref + ivr):
                        edge_from = pref + ivr
                    else:
                        edge_from = ivr


                    edge_to = pref + str(outv)

                    G.add_edge(edge_from, edge_to)

                else:
                    edge_from = str(v)
                    if isinstance(v, core.Literal):
                        id_name = next(id_names)
                        G.add_node(
                            pref + id_name,
                            label=id_name,
                            shape=v.aval.shape,
                            dtype=str(v.aval.dtype),
                            type='literal',
                            remat=remat,
                        )

                        edge_from = pref + id_name

                    edge_to = pref + str(outv)

                    G.add_edge(edge_from, edge_to)
            continue

        if str(eqn.primitive) in ['reshape', 'convert_element_type', 'broadcast_in_dim'] and len(eqn.invars) == 1:
            v = eqn.invars[0]

            if isinstance(v, core.Literal):
                id_name = next(id_names)
                G.add_node(
                    pref + id_name,
                    label=id_name,
                    shape=v.aval.shape,
                    dtype=str(v.aval.dtype),
                    type='literal',
                    remat=remat,
                )
                update_nodes[str(eqn.outvars[0])] = id_name
            else:
                iv_r = get_ivr(md['invars'], v)
                update_nodes[str(eqn.outvars[0])] = iv_r
            continue

        import uuid
        npref = ""
        if str(eqn.primitive) in ['custom_jvp_call', 'remat2', 'named_call']:
            if str(eqn.primitive) == 'remat2':
                sub_jaxpr = eqn.params['jaxpr']

            elif str(eqn.primitive) == 'custom_jvp_call':
                sub_jaxpr = eqn.params['call_jaxpr'].jaxpr
                npref = str(uuid.uuid4())[:8] + " "
            elif str(eqn.primitive) == 'named_call':
                sub_jaxpr = eqn.params['call_jaxpr']

            invar_dict = {}
            outvar_dict = {}

            for i_i, i_val in enumerate(sub_jaxpr.invars):
                i_v = str(i_val)

                if i_v in md['invars']:
                    invar_dict[i_v] = md['invars'][i_v]
                else:
                    invar_dict[i_v] = str(eqn.invars[i_i])

            for i_i, i_val in enumerate(sub_jaxpr.outvars):
                i_v = str(i_val)

                if i_v in md['outvars']:
                    outvar_dict[i_v] = md['outvars'][i_v]
                else:
                    outvar_dict[str(eqn.outvars[i_i])] = i_v


            invar_dict.update(md['invars'])
            outvar_dict.update(md['outvars'])

            G, id_names, update_nodes = _jaxpr_graph_g(
                sub_jaxpr,
                G=G,
                id_names=id_names,
                jtype=1,
                jpridx=jpridx,
                pref=npref,
                jaxpr_invars = jaxpr_invars,
                md={'invars': invar_dict, 'outvars': outvar_dict},
                remat=str(eqn.primitive) == 'remat2',
                update_nodes=update_nodes,
            )

        else:
            for v in eqn.invars:
                if isinstance(v, core.Literal):
                    G.add_node(
                        str(id(v.val)),
                        label=core.raise_to_shaped(core.get_aval(v.val)).str_short(),
                        shape=v.aval.shape,
                        dtype=str(v.aval.dtype),
                        type='literal',
                        remat=remat
                    )

            if eqn.primitive.multiple_results:
                id_name = next(id_names)
                node_id = str(eqn.primitive) + "_" + id_name
                G.add_node(
                    id_name,
                    label= node_id, #id_name,#str(eqn.primitive),
                    shape=eqn.outvars[0].aval.shape if len(eqn.outvars) > 0 else None,
                    dtype=str(eqn.outvars[0].aval.dtype) if len(eqn.outvars) > 0 else None,
                    type='op_node',
                    remat=remat
                )

                for i, v in enumerate(eqn.invars):
                    if jtype != 0 and v in jaxpr.invars:
                        iv_r = get_ivr(md['invars'], v)

                        ivr = get_ivr(update_nodes, iv_r)

                        if isinstance(v, core.Literal):
                            edge_from = str(id(v.val))
                        elif G.has_node(pref + ivr):
                            edge_from = pref + ivr
                        else:
                            edge_from = ivr

                        edge_to = id_name

                        G.add_edge(edge_from, edge_to)
                    else:
                        from_node = str(v)
                        if isinstance(v, core.Literal):
                            from_node = str(id(v.val))
                        else:
                            from_node = get_ivr(update_nodes, from_node)
                        G.add_edge(from_node, id_name)
                for v in eqn.outvars:
                    G.add_node(
                        pref + str(v),
                        label=v.aval.str_short(),
                        shape=v.aval.shape,
                        dtype=str(v.aval.dtype),
                        type='intermediate',
                        remat=remat
                    )
                    G.add_edge(id_name, pref + str(v))
            else:
                outv, = eqn.outvars
                G.add_node(
                    pref + str(outv),
                    label=str(eqn.primitive),
                    shape=outv.aval.shape,
                    dtype=str(outv.aval.dtype),
                    type='op_node',
                    remat=remat
                )

                for v in eqn.invars:
                    if jtype != 0 and v in jaxpr.invars:
                        iv_r = get_ivr(md['invars'], v)

                        ivr = get_ivr(update_nodes, iv_r)

                        if isinstance(v, core.Literal):
                            edge_from = str(id(v.val))
                        elif G.has_node(pref + ivr):
                            edge_from = pref + ivr
                        else:
                            edge_from = str(ivr)

                        edge_to = pref + str(outv)

                        G.add_edge(edge_from, edge_to)

                    else:
                        from_node = str(v)
                        if isinstance(v, core.Literal):
                            from_node = str(id(v.val))
                        else:
                            from_node = get_ivr(update_nodes, from_node)
                        G.add_edge(from_node, str(outv))

    for i, v in enumerate(jaxpr.outvars):
        if jtype != 0: # and G.has_node(str(md['outvars'][v])):
            vr = get_ivr(update_nodes, str(v))
            ovrs = get_ovrs(md['outvars'], str(v))
            for ovr in ovrs:
                if str(ovr) == str(vr):
                    continue

                if not G.has_node(str(ovr)):

                    update_nodes[ovr] = vr
        else:
            if not G.has_node(str(v)):
                G.add_node(
                    str(v),
                    label=str(v),
                    shape=v.aval.shape,
                    dtype=str(v.aval.dtype),
                    type='outvar',
                    remat=remat,
                )
                source_node = get_ivr(update_nodes, str(v))

                G.add_edge(source_node, str(v))
            else:
                vr = get_ivr(update_nodes, str(v))
                G.nodes[vr]['type'] = 'outvar'

    return (G, id_names, update_nodes)


def run_model(g, c, i, mesh_id, mm):
    res = run(g, c, i%2)
    mm[(mesh_id, i)] = res


def pred_latency(
    virtual_mesh: VirtualPhysicalMesh,
    submesh_choices: Sequence[Tuple[int]],
    autosharding_configs: Sequence[Sequence[Tuple[LogicalDeviceMesh, dict]]],
    layers: Sequence[JaxPipelineComputation],
    accumulator_mapping: Dict[Var, Var],
    acc_grad_invars: Sequence[Var],
    acc_grad_outvars: Sequence[Var],
    apply_grad_layers: Sequence[JaxPipelineComputation],
    apply_grad_global_info: Tuple,
    num_micro_batches: int,
    default_as_option: AutoShardingOption,
    stage_option: "AutoStageOption"
):
    cluster_size = virtual_mesh.num_devices
    num_autosharding_configs = len(autosharding_configs[0])
    mesh_models = {}
    
    for mesh_id, _ in reversed(list(enumerate(submesh_choices))):
        for as_id in range(num_autosharding_configs):
            if not os.path.exists(os.environ.get('SAVED_MODELS_DIR') + '/' + str(mesh_id) + '_' + str(as_id) + '.pth'):
                mesh_models[(mesh_id, as_id)] = None
                continue
            model_conf = create_model()                    
            model_conf['model'].load_state_dict(
                torch.load(os.environ.get('SAVED_MODELS_DIR') + '/' + str(mesh_id) + '_' + str(as_id) + '.pth')
            )
            mesh_models[(mesh_id, as_id)] = model_conf['model']
    
    mesh_profile_results = {}
    for mesh_id, submesh in reversed(list(enumerate(submesh_choices))):
        tic = time()
        num_hosts, num_devices_per_host = submesh
        if global_config.profile_with_whole_ray_cluster:
            whole_cluster_virtual_mesh = get_global_cluster(
            ).get_virtual_physical_mesh()
            sliced_virtual_meshes = (
                whole_cluster_virtual_mesh.slice_profiling_submeshes(
                    num_hosts, num_devices_per_host))
        else:
            sliced_virtual_meshes = virtual_mesh.slice_profiling_submeshes(
                num_hosts, num_devices_per_host)
        
        timers("model_pred_pred_gen").start()
        print(stage_option)
        
        sts = ([(i[0], i[-1]) for i in stage_option.forward_stage_layer_ids])
        stages = process_training_stages_2d(sts, layers,
                                accumulator_mapping,
                                acc_grad_invars,
                                acc_grad_outvars,
                                apply_grad_layers,
                                apply_grad_global_info,
                                mesh_id,
                                autosharding_configs[mesh_id],
                                sliced_virtual_meshes[0].num_devices,
                                cluster_size,
                                compile=True)
        
        timers("model_pred_pred_gen").stop()

        timers("model_pred_pred_pred").start()
        fwg = {}
        for i, st in enumerate(stages):
            fwg[(st[0][0], st[0][1])] = st[3][0]

        gphs = []
        for i, st in fwg.items():
            jpc = JaxPipelineComputation.from_closed_jaxpr('test', st)
            G, _, _ = _jaxpr_graph_g(jpc, jpridx=i)
            gphs.append(G)

        for asc in range(num_autosharding_configs):
            if mesh_models[(mesh_id, asc)] == None:
                continue
            res = pred(mesh_models[(mesh_id, asc)], gphs)
            print(res)

            for i, val in enumerate(res):
                layers_tup = list(fwg.keys())[i]
                mesh_profile_results[(layers_tup[0], layers_tup[1], mesh_id, asc)] = val#/100

            toc = time()
            print(f"Profiling for submesh {mesh_id} {submesh} takes {toc - tic:.2f}"
                f" seconds")
            print("-" * 50)
        timers("model_pred_pred_pred").stop()
    
    print(mesh_profile_results)
    
    stage_ids = [(i[0], i[-1]) for i in stage_option.forward_stage_layer_ids]
    mesh_ids = [submesh_choices.index(i) for i in stage_option.submesh_physical_shapes]
    asids = [get_as_index(autosharding_configs[i],
                          stage_option.submesh_autosharding_option_dicts[i],
                          stage_option.submesh_logical_shapes[j]) for j, i in enumerate(mesh_ids)]

    stage_lats = [mesh_profile_results[(
        stage_ids[i][0],
        stage_ids[i][1],
        mesh_ids[i],
        asids[i] 
    )] for i in range(len(stage_ids))]
    
    total_lat = max(stage_lats) * (num_micro_batches - 1) + sum(stage_lats)
    
    return total_lat

def get_as_index(autosharding_configs, asopts, logical_shape):
    asids = None

    for i, asc in enumerate(autosharding_configs):
        if asc is None:
            continue
        print(asc[1], asc[0].shape, asopts)
        if asc[1] == asopts and asc[0].shape == logical_shape:
            asids = i
            break
    return asids


# Including the few-shot learning method
def get_submesh_models_fsl(
        virtual_mesh: VirtualPhysicalMesh,
        submesh_choices: Sequence[Tuple[int]],
        autosharding_configs: Sequence[Sequence[Tuple[LogicalDeviceMesh, dict]]],
        layers: Sequence[JaxPipelineComputation],
        accumulator_mapping: Dict[Var, Var],
        acc_grad_invars: Sequence[Var],
        acc_grad_outvars: Sequence[Var],
        apply_grad_layers: Sequence[JaxPipelineComputation],
        apply_grad_global_info: Tuple,
        num_micro_batches: int,
        default_as_option: AutoShardingOption,
        auto_stage_option: "AutoStageOption"):

    cluster_size = virtual_mesh.num_devices
    layer_flops_prefix_sum = _get_layer_flops_prefix_sum(layers)
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    num_submesh_choices = len(submesh_choices)
    num_autosharding_configs = len(autosharding_configs[0])

    if auto_stage_option.cached_profile_result is not None:
        with open(auto_stage_option.cached_profile_result, "rb") as f:
            profile_results = pickle.load(f)
    else:
        profile_results = {}
    print("-" * 20 + " Automatic stage clustering " + "-" * 20)
    print(f"submesh_choices: {submesh_choices}")

    mesh_models = {}

    computed_graphs = {}


    # FIXME: reading all the cost data before for no profile tests
    # with open(auto_stage_option.full_data_file, 'rb') as f:
    #     all_costs_data = pickle.load(f)
        # print(all_costs_data)
        # cost_data={}
        # if auto_stage_option.full_data_file.startswith('gpt'):
        #     for k in all_costs_data:
        #         if (k[2], k[3]) not in cost_data:
        #             cost_data[(k[2], k[3])] = {}
        #         cost_data[(k[2], k[3])][(k[0], k[1])] = all_costs_data[k]
        # all_costs_data = cost_data
    

    few_shot_values = {}

    selected_keys = []

    print(auto_stage_option)
    for mesh_id, submesh in reversed(list(enumerate(submesh_choices))):
        num_hosts, num_devices_per_host = submesh

        whole_cluster_virtual_mesh = get_global_cluster().get_virtual_physical_mesh()
        sliced_virtual_meshes = (
            whole_cluster_virtual_mesh.slice_profiling_submeshes(
                num_hosts, num_devices_per_host))

        # DATAGEN: comment this part
        if sliced_virtual_meshes[0].num_devices == cluster_size:
            for asidx in range(num_autosharding_configs):
                mesh_models[(mesh_id, asidx)] = None
            continue

        print(f"- Trying fs learning for submesh {mesh_id} {submesh}:")
        

        ## Large stages on some models require, very large memory to for inter-operator optimizer parallely
        ## So divide it into chunks based on size and then compile each chunk parallely
        
        st_ends = get_training_stages_to_profile(layers, prob=0.1, reduce=1)
        selected_keys = st_ends
        
        stages = process_training_stages_2d(st_ends, layers,
                            accumulator_mapping,
                            acc_grad_invars,
                            acc_grad_outvars,
                            apply_grad_layers,
                            apply_grad_global_info,
                            mesh_id,
                            autosharding_configs[mesh_id],
                            sliced_virtual_meshes[0].num_devices,
                            cluster_size,
                            # FIXME: change for no profile
                            compile=True)
        
        timers("train_profile").start()
        # FIXME: comment for no profile
        profile_results = distributed_profile_on_mesh(
            stages, sliced_virtual_meshes, num_micro_batches, default_as_option,
            auto_stage_option, profile_results, op_type="prof")
        timers("train_profile").stop()
        

        print("Profile complete for submesh ", submesh)
        costs = {}

        # (L, L, S, C)
        # FIXME: change for no profile
        for k, pr in profile_results.items():
            cost = 0
            for mr in pr.module_profile_results:
                cost += mr.compute_cost
            costs[k] = cost

        fwg = {}
        for i, st in enumerate(stages):
            fwg[(st[0][0], st[0][1])] = st[3][0]

        cts = []
        for _ in range(num_autosharding_configs):
            cts.append([])

        gph_dict = {}
        count = 0
        for as_id, as_conf in enumerate(autosharding_configs[mesh_id]):
            if as_conf == None:
                continue
            for i, st in fwg.items():
                if i not in computed_graphs:
                    # raise Exception("Error graph not found!")
                    jpc = JaxPipelineComputation.from_closed_jaxpr('test', st)
                    G,_, _ = _jaxpr_graph_g(jpc, jpridx=i, update_nodes={})

                    computed_graphs[i] = G

                pr_key = (i[0], i[1], mesh_id, as_id)
                if pr_key not in costs:
                    continue
                gph_dict[i] = costs[pr_key]

                count += 1
            few_shot_values[(mesh_id, as_id)] = gph_dict

    complete_measurements = []

    # DONE MEASURING 10% FOR ALL COMBINATIONS
    # NOW TRY TO USE FSL ON THE ALL
    mesh_models = {}
    mesh_profile_results =  {}

    print("Starting the mapping process")
    for mesh_id, submesh in list(enumerate(submesh_choices)):
        print("FSL for mesh: ", mesh_id)
        for as_id, as_conf in enumerate(autosharding_configs[mesh_id]):
            if as_conf == None:
                continue

            num_hosts, num_devices_per_host = submesh

            whole_cluster_virtual_mesh = get_global_cluster(
            ).get_virtual_physical_mesh()
            sliced_virtual_meshes = (
                whole_cluster_virtual_mesh.slice_profiling_submeshes(
                    num_hosts, num_devices_per_host))

            # DATAGEN: comment this part
            if sliced_virtual_meshes[0].num_devices == cluster_size:
                for asidx in range(num_autosharding_configs):
                    mesh_models[(mesh_id, asidx)] = None
                continue
            
            print("FSL for as_id: ", as_id)

            if auto_stage_option.use_history_data:
                history_measurements = None
                with open(auto_stage_option.historical_data_path, 'rb') as f:
                    history_measurements = pickle.load(f)
                    
                for c in history_measurements:
                    newc = (c[0] + 1000, c[1])
                    few_shot_values[newc] = history_measurements[c]
                    complete_measurements.append(newc)

            fs_ok = False

            if len(complete_measurements) != 0:
                
                if auto_stage_option.use_history_data:
                     with open(auto_stage_option.historical_data_path, 'rb') as f:
                         history_measurements = pickle.load(f)
                         
                min_err = 1e9
                min_err_comb = None
                
                if len(complete_measurements) > 0:
                    for comb in complete_measurements:
                        err, xs, xs_full, pred_ys = rank_get_single(few_shot_values, comb, (mesh_id, as_id))
                        if min_err > err:
                            min_err = err
                            min_err_comb = (xs_full, pred_ys)

                        
                # print("mec", min_err_comb)
                print("min_err: ", min_err)
                print("fs cutoff: ", auto_stage_option.fs_cutoff)
                if min_err < auto_stage_option.fs_cutoff:
                    if min_err_comb != None:
                        for i, x in enumerate(min_err_comb[0]):
                            if x not in few_shot_values[(mesh_id, as_id)]:
                                few_shot_values[(mesh_id, as_id)][x] = min_err_comb[1][i]
                            # if i < 20:
                                # print(few_shot_values[(mesh_id, as_id)][x])
                            mesh_profile_results[(x[0], x[1], mesh_id, as_id)] = few_shot_values[(mesh_id, as_id)][x]
                    else:
                        raise Exception('failed!')

                    complete_measurements.append((mesh_id, as_id))
                    fs_ok = True
            if not fs_ok:
                print("Failed FS learn")
                print(f"- Trying fs learning for submesh {mesh_id} {submesh} {as_id}:")
                
                # for i in range(32):
                #     for j in range(i):
                #         s = (j, i)
                #         mesh_profile_results[(j, i, mesh_id, as_id)] = all_costs_data[(mesh_id, as_id)][s]
                #         few_shot_values[(mesh_id, as_id)][s] = all_costs_data[(mesh_id, as_id)][s]
                # complete_measurements.append((mesh_id, as_id))
                # continue
                
                
                
                import time
                sttime = time.time()
                
                stages = generate_training_stages_2d(
                    layers, layer_flops_prefix_sum, accumulator_mapping,
                    acc_grad_invars, acc_grad_outvars, apply_grad_layers,
                    apply_grad_global_info, mesh_id,
                    [autosharding_configs[mesh_id][as_id]],
                    sliced_virtual_meshes[0].num_devices, cluster_size,
                    #FIXME: change for no profile
                    auto_stage_option.stage_imbalance_tolerance, compile=True, prob=auto_stage_option.profile_amt - 0.1, reduce=0,
                    exclude=selected_keys
                )

                # profile_results = {}
                profile_results = distributed_profile_on_mesh(
                    stages, sliced_virtual_meshes, num_micro_batches, default_as_option,
                    auto_stage_option, profile_results, op_type="prof")

                edtime = time.time()
                
                print("Time for profile: ", edtime, sttime)
                costs = {}
                
                # FIXME: change for no profile
                for k, pr in profile_results.items():
                    cost = 0
                    for mr in pr.module_profile_results:
                        cost += mr.compute_cost
                    costs[(k[0], k[1], k[2], as_id)] = cost
                

                print("New profiled costs")
                # print(costs)
                fwg = {}
                for i, st in enumerate(stages):
                    fwg[(st[0][0], st[0][1])] = st[3][0]

                cts = []
                for _ in range(num_autosharding_configs):
                    cts.append([])

                gph_dict = {}
                count = 0
                for i, st in fwg.items():
                    if i not in computed_graphs:
                        jpc = JaxPipelineComputation.from_closed_jaxpr('test', st)
                        G,_, _ = _jaxpr_graph_g(jpc, jpridx=i, update_nodes={})

                        computed_graphs[i] = G

                    pr_key = (i[0], i[1], mesh_id, as_id)
                    gph_dict[i] = costs[pr_key]

                    count += 1

                few_shot_values[(mesh_id, as_id)] = {**gph_dict, **few_shot_values[(mesh_id, as_id)]}

                # print(few_shot_values.keys())
                # for k in few_shot_values:
                #     print(k, len(list(few_shot_values[k].keys())))

                t_graphs = []
                t_targs = []

                for comb, cost in few_shot_values[(mesh_id, as_id)].items():
                    t_graphs.append(computed_graphs[comb])
                    t_targs.append(cost)

                timers("stage-construction-graph-train").start()
                res = run(t_graphs, t_targs)
                timers("stage-construction-graph-train").stop()


                mesh_models[(mesh_id, as_id)] = res['model']
                # print(res)

                gphs = []
                for k in computed_graphs:
                    gphs.append(computed_graphs[k])

                timers("stage-construction-graph-infer").start()
                res = pred(mesh_models[(mesh_id, as_id)], gphs)
                timers("stage-construction-graph-infer").stop()
                gkys = list(computed_graphs.keys())
                for i, val in enumerate(res):
                    layers_tup = gkys[i]

                    mesh_profile_results[(layers_tup[0], layers_tup[1], mesh_id, as_id)] = val#/100



                for key, val in mesh_profile_results.items():
                    if key[2] == mesh_id and key[3] == as_id:
                        if ((key[0], key[1])) not in few_shot_values[(mesh_id, as_id)]:
                            few_shot_values[(mesh_id, as_id)][(key[0], key[1])] = val

                complete_measurements.append((mesh_id, as_id))
    return mesh_profile_results

def rank_get_single(costs_raw, ref, targ):
    def func(X, a, b, c, d, e): #, f, g, h, i):
        x, y = X
        return a + y * (b + c * np.power(x, 1) + d * np.power(x, 2) + e * np.power(x, 3))

    # sorted dictionary of ref lats
    sorted_ref = {k: v for k, v in sorted(costs_raw[ref].items(), key=lambda item: item[1])}

    # keys of sorted lats
    sorted_ref_keys = list(sorted_ref.keys())

    # print(sorted_ref_keys)
    def filterfun(v):
        if v not in costs_raw[targ]:
            return False
        if costs_raw[ref][v] + costs_raw[targ][v] == np.inf:
            return False
        if costs_raw[ref][v] * costs_raw[targ][v] == 0:
            return False
        return True

    xs_full = [i for i in range(len(sorted_ref_keys))]

    sorted_targ_keys = list(filter(filterfun, sorted_ref_keys))

    #dividing target values into train and test
    total = len(sorted_targ_keys)
    test_idx = list(range(0, total,3))
    targ_test_keys = [sorted_targ_keys[i] for i in test_idx]
    sorted_targ_keys = [i for i in sorted_targ_keys if i not in targ_test_keys]

    targ_ys_full = [costs_raw[targ][i] for i in sorted_targ_keys]
    ref_ys_full = [costs_raw[ref][i] for i in sorted_ref_keys]
    ref_ys = np.array([costs_raw[ref][i] for i in sorted_targ_keys])
    xs = np.array([sorted_ref_keys.index(ss) for ss in sorted_targ_keys])
    targ_ys = np.array([costs_raw[targ][i] for i in sorted_targ_keys])

    popt, pcov = curve_fit(func, (xs, ref_ys), targ_ys)

    pred_ys = func(
        (
            [sorted_ref_keys.index(ss) for ss in targ_test_keys], # xs_full,
            [costs_raw[ref][i] for i in targ_test_keys]# ref_ys_full
        ), *popt)

    pred_ys[pred_ys<0] = 0

    pred_ys_full = func((xs_full, ref_ys_full), *popt)
    pred_ys_full[pred_ys_full<0] = 0

    test_targ_ys = [costs_raw[targ][i] for i in targ_test_keys]

    diff = np.array(pred_ys - test_targ_ys)
    diff_norm = (np.abs(diff) / test_targ_ys) * 100

    errs = np.mean(diff_norm)

    return errs, sorted_targ_keys, sorted_ref_keys, pred_ys_full

def get_compute_cost_pred(
        virtual_mesh: VirtualPhysicalMesh,
        submesh_choices: Sequence[Tuple[int]],
        autosharding_configs: Sequence[Sequence[Tuple[LogicalDeviceMesh,
                                                      dict]]],
        layers: Sequence[JaxPipelineComputation],
        accumulator_mapping: Dict[Var, Var],
        acc_grad_invars: Sequence[Var],
        acc_grad_outvars: Sequence[Var],
        apply_grad_layers: Sequence[JaxPipelineComputation],
        apply_grad_global_info: Tuple,
        num_micro_batches: int,
        default_as_option: AutoShardingOption,
        auto_stage_option: "AutoStageOption",
        inference_mode: bool = False,
        mesh_models: any = []):
    cluster_size = virtual_mesh.num_devices

    layer_flops_prefix_sum = _get_layer_flops_prefix_sum(layers)
    # autosharding_configs = [asc[:1] for asc in autosharding_configs]
    if inference_mode:
        num_layers = len(layers)
    else:
        assert len(layers) % 2 == 0
        num_layers = len(layers) // 2
    num_submesh_choices = len(submesh_choices)
    num_autosharding_configs = len(autosharding_configs[0])

    if auto_stage_option.cached_profile_result is not None:
        with open(auto_stage_option.cached_profile_result, "rb") as f:
            profile_results = pickle.load(f)
    else:
        profile_results = {}
    print("-" * 20 + " Automatic stage clustering " + "-" * 20)
    print(f"submesh_choices: {submesh_choices}")

    mesh_profile_results =  {}
    
    gphs = []  
    if not auto_stage_option.use_fsl:
        # Reverse submesh_choices to test larger meshes first
        for mesh_id, submesh in reversed(list(enumerate(submesh_choices))):
            print(f"- Profiling for submesh {mesh_id} {submesh}:")

            num_hosts, num_devices_per_host = submesh
            tic = time()
            if global_config.profile_with_whole_ray_cluster:
                whole_cluster_virtual_mesh = get_global_cluster(
                ).get_virtual_physical_mesh()
                sliced_virtual_meshes = (
                    whole_cluster_virtual_mesh.slice_profiling_submeshes(
                        num_hosts, num_devices_per_host))
            else:
                sliced_virtual_meshes = virtual_mesh.slice_profiling_submeshes(
                    num_hosts, num_devices_per_host)


            if (mesh_id, 0) not in mesh_models or mesh_models[(mesh_id, 0)] == None:
                if os.environ.get('SAVED_MODELS_DIR'):
                    with open(f'{os.environ.get("SAVED_MODELS_DIR")}/full_mesh_results.pkl', 'rb') as f:
                        profile_results = pickle.load(f)
                else:
                    timers("model_pred_gen").start()
                    stages = generate_training_stages_2d(
                        layers, layer_flops_prefix_sum, accumulator_mapping,
                        acc_grad_invars, acc_grad_outvars, apply_grad_layers,
                        apply_grad_global_info, mesh_id,
                        autosharding_configs[mesh_id],
                        sliced_virtual_meshes[0].num_devices, cluster_size,
                        auto_stage_option.stage_imbalance_tolerance, compile=True)

                    timers("model_pred_gen").stop()

                    timers("model_pred_profile").start()

                    profile_results = distributed_profile_on_mesh(
                        stages, sliced_virtual_meshes, num_micro_batches, default_as_option,
                        auto_stage_option, profile_results, op_type="pred")

                    timers("model_pred_profile").stop()
                    if os.environ.get('SAVE_MODEL_DIR') is not None:
                        Path(os.environ.get('SAVE_MODEL_DIR')).mkdir(parents=True, exist_ok=True)
                        with open(f'{os.environ.get("SAVE_MODEL_DIR")}/full_mesh_results.pkl', 'wb') as f:
                            pickle.dump(profile_results, f)
                    

                for idx in profile_results:
                    pr = profile_results[idx]
                    mesh_profile_results[idx] = np.inf 
                    mesh_profile_results[idx] =sum(result.compute_cost for result in pr.module_profile_results)
                    
                print(profile_results)
                toc = time()
                print(f"Profiling for submesh {mesh_id} {submesh} takes {toc - tic:.2f}"
                    f" seconds")
                print("-" * 50)
            else:
                if len(gphs) == 0:
                    timers("model_pred_pred_gen").start()    
                    stages = generate_training_stages_2d(
                        layers, layer_flops_prefix_sum, accumulator_mapping,
                        acc_grad_invars, acc_grad_outvars, apply_grad_layers,
                        apply_grad_global_info, mesh_id,
                        autosharding_configs[mesh_id],
                        sliced_virtual_meshes[0].num_devices, cluster_size,
                        auto_stage_option.stage_imbalance_tolerance, compile=False, reduce=0)

                    print(len(stages))
                    fwg = {}
                    for i, st in enumerate(stages):
                        fwg[(st[0][0], st[0][1])] = st[3][0]

                    for i, st in fwg.items():
                        jpc = JaxPipelineComputation.from_closed_jaxpr('test', st)
                        G, _, _ = _jaxpr_graph_g(jpc, jpridx=i)
                        gphs.append(G)
                    print(len(gphs))
                    timers("model_pred_pred_gen").stop()
                
                # stgs = [s[:3] for s in stages]
                # check_profile_results_consistent(stgs, profile_results)
                timers("model_pred_pred_pred").start()
                for asc in range(num_autosharding_configs):
                    
                    if mesh_models[(mesh_id, asc)] == None:
                        continue
                    res = pred(mesh_models[(mesh_id, asc)], gphs)

                    for i, val in enumerate(res):
                        layers_tup = list(fwg.keys())[i]
                        mesh_profile_results[(layers_tup[0], layers_tup[1], mesh_id, asc)] = val#/100

                    print("-" * 50)
                toc = time()
                print(f"Profiling for submesh {mesh_id} {submesh} takes {toc - tic:.2f}"
                        f" seconds")
                timers("model_pred_pred_pred").stop()

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        profile_result_file_name = (f"profile-results-{timestamp}.npy")
        np.save(profile_result_file_name, profile_results)
        global last_compute_cost_file_name
        last_compute_cost_file_name = profile_result_file_name
        print(f"Profile result saved to: {profile_result_file_name}")
        print("-" * 70)
    else:
        mesh_profile_results = mesh_models
    
    all_compute_cost = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        np.inf,
        dtype=np.float64)

    all_max_n_succ_stages = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        np.inf,
        dtype=np.float64)

    for index in np.ndindex(num_layers, num_layers, num_submesh_choices,
                            num_autosharding_configs):
        if index not in mesh_profile_results:
            continue
        all_compute_cost[index] = mesh_profile_results[index]

    return all_compute_cost, all_max_n_succ_stages

def profile_all(stages, compiled_outputs: Sequence[CompileOutput], meshes,
                num_micro_batches, auto_stage_option, profile_results):
    """Profile all compiled outputs on given meshes.

    This function launches a profile worker pool and submits given tasks.
    """
    placement_group = retrieve_placement_group()

    if auto_stage_option.use_hlo_cost_model:
        num_cpus = int(
            min(max(ray.available_resources()["CPU"] // 3, 1), len(stages)))
        mesh_num_devices = meshes[0].num_devices
        prof_database = ProfilingResultDatabase()
        prof_database.load(auto_stage_option.profiling_database_filename)
        prof_result = prof_database.query("default", meshes[0].shape)
        profile_workers = HloCostModelProfileWorkerPool(num_cpus,
                                                        placement_group,
                                                        prof_result,
                                                        mesh_num_devices,
                                                        num_micro_batches)
    else:
        profile_workers = ProfileWorkerPool(meshes, placement_group)

    successful_compile_ct = 0
    for i, (compiled_output, stage) in enumerate(zip(compiled_outputs, stages)):
        if compiled_output is None:
            continue
        stage_idx, stage_config, _ = stage

        for module_id, (acc_grad_module, profile_config) in enumerate(
                zip(compiled_output.acc_grad_module_compile_outputs,
                    stage_config.module_profile_configs)):
            if profile_results[stage_idx].is_module_profiled(module_id):
                continue
            profile_workers.submit(lambda w, v: w.profile.remote(*v),
                                   ((i, module_id), acc_grad_module,
                                    compiled_output.stage_plan, profile_config))
            successful_compile_ct += 1

    pbar = tqdm.tqdm(range(successful_compile_ct))
    for _ in pbar:
        try:
            ((i, module_id),
             *module_raw_result) = profile_workers.get_next_unordered()
        except TimeoutError:
            profile_workers.shutdown(force=True)
            logger.warning("After waiting for too long, "
                           "all profile workers are forcely killed")
            return profile_results
        except (RuntimeError, RayActorError):
            profile_workers.shutdown(force=True)
            logger.warning("Meet unexpected error, "
                           "all profile workers are forcely killed")
            return profile_results
        stage_idx, stage_config, _ = stages[i]
        stage_compile_output = compiled_outputs[i]
        module_profile_result = generate_module_profile_result(
            module_raw_result, stage_config.module_profile_configs[module_id],
            stage_compile_output.acc_grad_module_compile_outputs[module_id],
            stage_compile_output.stage_plan.logical_mesh_shape)
        pbar.write(f"result[{stage_idx}, {module_id}] "
                   f"= {module_profile_result}")
        profile_results[stage_idx].add_module_profile_result(
            module_id, module_profile_result)
    profile_workers.shutdown()
    return profile_results

def distributed_profile_on_mesh(stages, meshes: Sequence[VirtualPhysicalMesh],
                                num_micro_batches, default_as_option,
                                auto_stage_option, profile_results, op_type="prof"):
    op_type = "-" + op_type
    timers("stage-construction-compilation"+op_type).start()

    stages = [s[:3] for s in stages]
    
    if len(stages) == 0:
        # Suspend timers
        timers("stage-construction-compilation"+op_type).stop()
        return profile_results

    print("- Compile all stages")
    compiled_outputs = []
    try:
        limit = auto_stage_option.compile_limit
        stage_chunks = []
        stsum = 0
        current_chunk = []
        for se in stages:
            diff = se[0][1] - se[0][0]
            if stsum + diff > limit:
              stsum = 0
              stage_chunks.append(current_chunk)
              current_chunk = []

            stsum += diff
            current_chunk.append(se)

        if len(current_chunk) != 0:
            stage_chunks.append(current_chunk)
        
        for chunk in stage_chunks:            
            compiled_outputs += compile_all(chunk, num_micro_batches,
                                       default_as_option, profile_results)          
    except RayActorError as e:
        logger.warning(f"Compilation fatal error: {e}")
        timers("stage-construction-compilation"+op_type).stop()
        return profile_results
    timers("stage-construction-compilation"+op_type).stop()

    print("- Profile all stages")
    timers("stage-construction-profiling"+op_type).start()
    
    
    profile_results = profile_all(
            stages,
            compiled_outputs, meshes,
            num_micro_batches, auto_stage_option,
            profile_results)

    timers("stage-construction-profiling"+op_type).stop()
    return profile_results

def generate_stage_info(all_layers, selected_indices,
                        global_accumulator_mapping, acc_grad_invars,
                        acc_grad_outvars, name, apply_grad_layers,
                        apply_grad_info, compile=True):
    """Combine selected layers together for profiling."""
    modules = []
    module_accumulator_mappings = []
    module_required_outvars = []
    for layer_indices in selected_indices:
        module, module_accumulator_mapping, required_outvars = (
            select_module_layers(all_layers, layer_indices,
                                 global_accumulator_mapping, acc_grad_outvars))
        modules.append(module)
        module_accumulator_mappings.append(module_accumulator_mapping)
        module_required_outvars.append(required_outvars)

    n_modules = len(modules)
    module_jaxprs = [
        [layer.closed_jaxpr() for layer in layers] for layers in modules
    ]

    module_names = [f"{name}_acc_grad_{i}" for i in range(n_modules)]
    module_merged_jaxprs = []
    module_profile_configs = []

    all_modules_donation_mapping = {}
    all_modules_donate_invars = []
    all_modules_outvars = OrderedSet()
    all_modules_acc_grad_outvars_indices = []
    acc_grad_invars_set = OrderedSet(acc_grad_invars)
    acc_grad_outvars_set = OrderedSet(acc_grad_outvars)
    for module_name, jaxprs, accumulator_mapping, required_outvars in zip(
            module_names, module_jaxprs, module_accumulator_mappings,
            module_required_outvars):
        merged_jaxpr = merge_marked_jaxprs_with_named_call(
            jaxprs, required_outvars, accumulator_mapping, module_name)
        outvars_set = set(merged_jaxpr.jaxpr.outvars)
        is_donated = tuple(invar in accumulator_mapping and
                           accumulator_mapping[invar] in outvars_set
                           for invar in merged_jaxpr.jaxpr.invars)
        acc_grad_invars_indices = tuple(
            i for i, outvar in enumerate(merged_jaxpr.jaxpr.invars)
            if outvar in acc_grad_invars_set)
        acc_grad_outvars_indices = tuple(
            i for i, outvar in enumerate(merged_jaxpr.jaxpr.outvars)
            if outvar in acc_grad_outvars_set)
        invar_names = tuple(repr(var) for var in merged_jaxpr.jaxpr.invars)
        outvar_names = tuple(repr(var) for var in merged_jaxpr.jaxpr.outvars)
        invar_avals = tuple(var.aval for var in merged_jaxpr.jaxpr.invars)
        outvar_avals = tuple(var.aval for var in merged_jaxpr.jaxpr.outvars)
        profile_config = ModuleProfileConfig(invar_names, outvar_names,
                                             invar_avals, outvar_avals,
                                             is_donated,
                                             acc_grad_invars_indices,
                                             acc_grad_outvars_indices)
        module_merged_jaxprs.append(merged_jaxpr)
        module_profile_configs.append(profile_config)
        all_modules_donate_invars.append(is_donated)
        all_modules_donation_mapping.update(accumulator_mapping)
        all_modules_outvars.update(merged_jaxpr.jaxpr.outvars)
        all_modules_acc_grad_outvars_indices.append(acc_grad_outvars_indices)

    if len(apply_grad_layers) > 0:
        apply_grad_donation, apply_grad_outvars = apply_grad_info
        apply_grad_module_name = "_".join([name, APPLY_GRAD_MARKER_SUFFIX])
        merged_apply = merge_marked_jaxprs_with_named_call(
            [layer.closed_jaxpr() for layer in apply_grad_layers],
            apply_grad_outvars, apply_grad_donation, name + "_apply")
        outvars_set = set(merged_apply.jaxpr.outvars)
        is_donated = tuple(invar in apply_grad_donation and
                           apply_grad_donation[invar] in outvars_set
                           for invar in merged_apply.jaxpr.invars)
        apply_only_invars = OrderedSet(merged_apply.jaxpr.invars)
        for module_jaxpr in module_merged_jaxprs:
            apply_only_invars = apply_only_invars.difference(
                module_jaxpr.jaxpr.invars)
            apply_only_invars = apply_only_invars.difference(
                module_jaxpr.jaxpr.outvars)
        apply_info = ApplyGradConfig(merged_apply.jaxpr.invars,
                                     apply_only_invars)
        module_names.append(apply_grad_module_name)
        module_merged_jaxprs.append(merged_apply)
        all_modules_donate_invars.append(is_donated)
        all_modules_donation_mapping.update(apply_grad_donation)
        all_modules_outvars.update(merged_apply.jaxpr.outvars)
    else:
        apply_info = None

    all_modules_merged_jaxpr, all_modules_is_donated = (
        merge_unmarked_with_call(module_merged_jaxprs, module_names,
                                 all_modules_outvars,
                                 all_modules_donation_mapping))

    hlo = None
    if compile:
        hlo = jaxpr_to_hlo(name, all_modules_merged_jaxpr, all_modules_is_donated)
    compile_config = CompileConfig(hlo, module_names, all_modules_donate_invars,
                                   all_modules_acc_grad_outvars_indices)
    stage_config = StageConfig(n_modules, compile_config,
                               module_profile_configs, apply_info)
    return stage_config, module_merged_jaxprs
