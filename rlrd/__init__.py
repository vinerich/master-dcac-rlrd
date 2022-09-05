import atexit
import gc
import json
import os
import shutil
import tempfile
from os.path import exists
from random import randrange
from tempfile import mkdtemp

import pandas as pd
import yaml

from rlrd.util import partial, save_json, partial_to_dict, partial_from_dict, load_json, dump, load, git_info
from rlrd.training import Training
import rlrd.sac
import rlrd.sac_models_rd
import rlrd.dcac
import rlrd.dcac_models
import rlrd.envs

from gym.envs.registration import register

register(
    id='zinc-coating-v0',
    entry_point='zinc_coating.zinc_coating_environment:ZincCoatingV0',
)

def iterate_episodes(run_cls: type = Training, checkpoint_path: str = None):
    """Generator [1] yielding episode statistics (list of pd.Series) while running and checkpointing
    - run_cls: can by any callable that outputs an appropriate run object (e.g. has a 'run_epoch' method)

    [1] https://docs.python.org/3/howto/functional.html#generators
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        if not exists(checkpoint_path):
            print("=== specification ".ljust(70, "="))
            print(yaml.dump(partial_to_dict(run_cls), indent=3, default_flow_style=False, sort_keys=False), end="")
            run_instance = run_cls()
            dump(run_instance, checkpoint_path)
            print("")
        else:
            print("\ncontinuing...\n")

        run_instance = load(checkpoint_path)
        while run_instance.epoch < run_instance.epochs:
            # time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
            yield run_instance.run_epoch()  # yield stats data frame (this makes this function a generator)
            print("")
            dump(run_instance, checkpoint_path)

            # we delete and reload the run_instance from disk to ensure the exact same code runs regardless of interruptions
            del run_instance
            gc.collect()
            run_instance = load(checkpoint_path)

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def log_environment_variables():
    """add certain relevant environment variables to our config
    usage: `LOG_VARIABLES='HOME JOBID' python ...`
    """
    return {k: os.environ.get(k, '') for k in os.environ.get('LOG_VARIABLES', '').strip().split()}


def run(run_cls: type = Training, checkpoint_path: str = None):
    list(iterate_episodes(run_cls, checkpoint_path))


def run_wandb(entity, project, run_id, run_cls: type = Training, checkpoint_path: str = None):
    """run and save config and stats to https://wandb.com"""
    wandb_dir = mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
    import wandb
    config = partial_to_dict(run_cls)
    config['seed'] = config['seed'] or randrange(1, 1000000)  # if seed == 0 replace with random
    config['environ'] = log_environment_variables()
    config['git'] = git_info()
    resume = checkpoint_path and exists(checkpoint_path)
    wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
    for stats in iterate_episodes(run_cls, checkpoint_path):
        [wandb.log(json.loads(s.to_json())) for s in stats]


def run_fs(path: str, run_cls: type = Training):
    """run and save config and stats to `path` (with pickle)"""
    if not exists(path):
        os.mkdir(path)
    save_json(partial_to_dict(run_cls), path + '/spec.json')
    if not exists(path + '/stats'):
        dump(pd.DataFrame(), path + '/stats')
    for stats in iterate_episodes(run_cls, path + '/state'):
        dump(load(path + '/stats').append(stats, ignore_index=True),
             path + '/stats')  # concat with stats from previous episodes


# === specifications ===================================================================================================

DcacTest = partial(
    Training,
    epochs=1,
    rounds=5,
    steps=1000,
    Agent=partial(
        rlrd.dcac.Agent,
        device="cuda",
        rtac=False,  # set this to True for reverting to RTAC
        batchsize=64,
        start_training=50,
        Model=partial(
            rlrd.dcac_models.Mlp,
            act_delay=True,
            obs_delay=True)),
    Env=partial(
        rlrd.envs.RandomDelayEnv,
        id="Pendulum-v0",
        min_observation_delay=0,
        sup_observation_delay=2,
        min_action_delay=0,
        sup_action_delay=2,
        real_world_sampler=0),
)

DcacTraining = partial(
    Training,
    Agent=partial(
        rlrd.dcac.Agent,
        device="cpu",
        rtac=False,  # set this to True for reverting to RTAC
        batchsize=64,
        Model=partial(
            rlrd.dcac_models.Mlp,
            act_delay=True,
            obs_delay=True)),
    Env=partial(
        rlrd.envs.RandomDelayEnv,
        id="zinc-coating-v0",
        min_observation_delay=0,
        sup_observation_delay=1,
        min_action_delay=0,
        sup_action_delay=1,
        real_world_sampler=0),
)

DcacShortTimesteps = partial(  # works at 2/5 of the original Mujoco timescale
    DcacTraining,
    Env=partial(frame_skip=2),  # only works with Mujoco tasks (for now)
    steps=5000,
    Agent=partial(memory_size=2500000, training_steps=2 / 5, start_training=25000, discount=0.996, entropy_scale=2 / 5)
)

# To compare against SAC:
DelayedSacTest = partial(
    Training,
    epochs=1,
    rounds=5,
    steps=1000,
    Agent=partial(
        rlrd.sac.Agent,
        device="cuda",
        batchsize=64,
        start_training=50,
        Model=partial(
            rlrd.sac_models_rd.Mlp,
            act_delay=True,
            obs_delay=True),
        OutputNorm=partial(beta=0., zero_debias=False),
    ),
    Env=partial(
        rlrd.envs.RandomDelayEnv,
        id="Pendulum-v0",
        min_observation_delay=0,
        sup_observation_delay=2,
        min_action_delay=0,
        sup_action_delay=2,
    ),
)

DelayedSacTraining = partial(
    Training,
    Agent=partial(
        rlrd.sac.Agent,
        batchsize=128,
        Model=partial(
            rlrd.sac_models_rd.Mlp,
            act_delay=True,
            obs_delay=True),
        OutputNorm=partial(beta=0., zero_debias=False),
    ),
    Env=partial(
        rlrd.envs.RandomDelayEnv,
        id="Pendulum-v0",
        min_observation_delay=0,
        sup_observation_delay=1,
        min_action_delay=0,
        sup_action_delay=1,
    ),
)

DelayedSacShortTimesteps = partial(  # works at 2/5 of the original Mujoco timescale
    DelayedSacTraining,
    Env=partial(frame_skip=2),  # only works with Mujoco tasks (for now)
    steps=5000,
    Agent=partial(memory_size=2500000, training_steps=2 / 5, start_training=25000, discount=0.996, entropy_scale=2 / 5)
)


# === tests ============================================================================================================
if __name__ == "__main__":
    print("--- Now running SAC: ---")
    run(DelayedSacTest)
    print("--- Now running DCAC: ---")
    run(DcacTest)
