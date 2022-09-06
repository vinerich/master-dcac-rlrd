# from copy import deepcopy
# import pickle
from dataclasses import dataclass

import pandas as pd
# import numpy as np
from pandas import DataFrame, Timestamp

import rlrd.sac

from rlrd.testing import Test
from rlrd.util import pandas_dict, cached_property
from rlrd.wrappers import StatsWrapper
from rlrd.envs import GymEnv
# from dcac_python.batch_env import get_env_state

# import pybullet_envs

import numpy as np


@dataclass(eq=0)
class Training:
    Env: type = GymEnv
    Test: type = Test
    Agent: type = rlrd.sac.Agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2500  # number of steps per round, one step = environment step
    stats_window: int = None  # default = steps, should be at least as long as a single episode
    seed: int = 0  # seed is currently not used
    tag: str = ''  # for logging, e.g. allows to compare groups of runs
    np_stats = {}
    np_stats["timesteps"] = []
    np_stats["results"] = []
    np_stats["results_lows"] = []
    np_stats["results_highs"] = []
    np_stats["results_rollout"] = []
    np_stats["ep_lengths"] = []
    np_stats["train_time"] = []
    np_init = False

    def __post_init__(self):
        self.epoch = 0
        self.agent = self.Agent(self.Env)

    def run_epoch(self, log_path):
        stats = []
        state = None

        with StatsWrapper(self.Env(seed_val=self.seed + self.epoch), window=self.stats_window or self.steps) as env:
            for rnd in range(self.rounds):
                print(f"=== epoch {self.epoch + 1}/{self.epochs} ".ljust(20, '=') + f" round {rnd + 1}/{self.rounds} ".ljust(50, '='))

                t0 = pd.Timestamp.utcnow()
                stats_training = []

                # start test and run it in parallel to the training process
                test = self.Test(
                    Env=self.Env,
                    actor=self.agent.model,
                    steps=self.stats_window or self.steps,
                    base_seed=self.seed + self.epochs
                )

                for step in range(self.steps):
                    action, state, training_stats = self.agent.act(state, *env.transition, train=True)
                    stats_training += training_stats
                    env.step(action)

                stats += pandas_dict(
                    **env.stats(),
                    round_time=Timestamp.utcnow() - t0,
                    **test.stats().add_suffix("_test"),  # this blocks until the tests have finished
                    round_time_total=Timestamp.utcnow() - t0,
                    **DataFrame(stats_training).mean(skipna=True)
                ),

                print(stats[-1].add_prefix("  ").to_string(), '\n')

                #convert stats
                self.np_stats["results"].append([stats[-1].returns_test])
                self.np_stats["results_lows"].append([stats[-1].returns_lows_test])
                self.np_stats["results_highs"].append([stats[-1].returns_highs_test])
                self.np_stats["results_rollout"].append(stats[-1].returns)
                self.np_stats["ep_lengths"].append(stats[-1].episode_length)
                if self.np_init:
                    self.np_stats["timesteps"].append(self.np_stats["timesteps"][-1] + stats[-1].episode_length)
                    self.np_stats["train_time"].append(self.np_stats["train_time"][-1] + stats[-1].round_time.total_seconds())
                else:
                    self.np_stats["timesteps"].append(stats[-1].episode_length)
                    self.np_stats["train_time"].append(stats[-1].round_time.total_seconds())
                    self.np_init = True

                np.savez(
                    log_path,
                    timesteps=self.np_stats["timesteps"],
                    results=self.np_stats["results"],
                    results_lows=self.np_stats["results_lows"],
                    results_highs=self.np_stats["results_highs"],
                    results_rollout=self.np_stats["results_rollout"],
                    ep_lengths=self.np_stats["ep_lengths"],
                    # lows=stats.returns,
                    # highs=self.evaluation_highs,
                    train_time=self.np_stats["train_time"]
                )

        self.epoch += 1
        return stats
