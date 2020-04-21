# Delayed RTAC models

import gym
import torch
from agents.nn import TanhNormalLayer
from agents.rtac_models import DoubleActorModule
from torch.nn import Linear
from torch.nn import functional as F

from agents.sac_models import ActorModule


# TODO: Add separate mlp model
# TODO: WIP


class DelayedBranchedMlp(ActorModule):
	def __init__(self, observation_space, action_space, hidden_units: int = 256):
		super().__init__()
		assert isinstance(observation_space, gym.spaces.Tuple)
		# TODO: check that it is actually an instance of:
		# Tuple((
		# 	obs_space,  # most recent observation
		# 	Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
		# 	Discrete(obs_delay_range.stop),  # observation delay int64
		# 	Discrete(act_delay_range.stop),  # action delay int64
		# ))

		self.obs_dim = observation_space[0].shape[0]
		self.buf_size = len(observation_space[1])
		self.act_dim = observation_space[1][0].shape[0]

		self.lin_obs = Linear(self.obs_dim + self.buf_size, hidden_units)  # TODO: find a better solution
		self.lin_act = Linear(self.act_dim * self.buf_size + self.buf_size, hidden_units)  # TODO: find a better solution
		self.lin_merged = Linear(2 * hidden_units, hidden_units)

		self.critic_layer = Linear(hidden_units, 2)  # predict future reward and entropy separately
		self.actor_layer = TanhNormalLayer(hidden_units, action_space.shape[0])
		self.critic_output_layers = (self.critic_layer,)

	def actor(self, x):
		return self(x)[0]

	def forward(self, x):
		assert isinstance(x, tuple)
		# TODO: check that x is actually in:
		# Tuple((
		# 	obs_space,  # most recent observation
		# 	Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
		# 	Discrete(obs_delay_range.stop),  # observation delay int64
		# 	Discrete(act_delay_range.stop),  # action delay int64
		# ))

		# TODO: WIP

		print(f"DEBUG:x[0].shape:{x[0].shape}")
		obs = x[0]
		print(f"DEBUG: len(x[1]):{len(x[1])}")
		print(f"DEBUG:x[2]:{x[2]}")
		print(f"DEBUG:x[3]:{x[3]}")
		act_buf = torch.cat(x[1], dim=1)
		obs_del = x[2]
		act_del = x[3]

		batch_size = obs.shape[0]
		obs_one_hot = torch.zeros(batch_size, self.buf_size, device=self.device).scatter_(1, obs_del.unsqueeze(1), 1.0)  # TODO: check that scatter_ doesn't create a [1.0] tensor on CPU
		act_one_hot = torch.zeros(batch_size, self.buf_size, device=self.device).scatter_(1, act_del.unsqueeze(1), 1.0)  # TODO: check that scatter_ doesn't create a [1.0] tensor on CPU

		# print(f"DEBUG:obs:{obs}")
		# print(f"DEBUG:act_buf:{act_buf}")
		# print(f"DEBUG:obs_del:{obs_del}")
		# print(f"DEBUG:act_del:{act_del}")
		# print(f"DEBUG:obs_one_hot:{obs_one_hot}")
		# print(f"DEBUG:act_one_hot:{act_one_hot}")

		input_obs = torch.cat((obs, obs_one_hot), dim=1)
		input_act = torch.cat((act_buf, act_one_hot), dim=1)

		# print(f"DEBUG:input_obs.shape:{input_obs.shape}")
		# print(f"DEBUG:input_act.shape:{input_act.shape}")

		h_obs = F.relu(self.lin_obs(input_obs))
		h_act = F.relu(self.lin_act(input_act))

		# print(f"DEBUG:h_obs.shape:{h_obs.shape}")
		# print(f"DEBUG:h_act.shape:{h_act.shape}")

		h = torch.cat((h_obs, h_act), dim=1)

		# print(f"DEBUG:h.shape:{h.shape}")

		h = F.relu(self.lin_merged(h))

		# print(f"DEBUG:h.shape:{h.shape}")

		v = self.critic_layer(h)
		action_distribution = self.actor_layer(h)

		# assert False, "DEBUG: Went this far"

		# print(f"DEBUG: ouput: action_distribution:{action_distribution}, v:{v}, h.shape:{h.shape}")

		return action_distribution, (v,), (h,)


class DelayedBranchedMlpDouble(DoubleActorModule):
	def __init__(self, observation_space, action_space, hidden_units: int = 256):
		super().__init__()
		self.a = DelayedBranchedMlp(observation_space, action_space, hidden_units=hidden_units)
		self.b = DelayedBranchedMlp(observation_space, action_space, hidden_units=hidden_units)