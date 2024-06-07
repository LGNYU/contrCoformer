#!/usr/bin/env python3

import warnings
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import numpy as np

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder
from rewarder import optimal_transport_plan, cosine_distance, euclidean_distance

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )
        if self.cfg.suite.name == "dmc":
            self.cfg.suite.task_make_fn.max_action_dim = (
                self.expert_replay_loader.dataset._max_action_dim
            )
        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = len(self.env)
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def compute_distance(self, traj1, traj2, task_emb1, task_emb2, metric):
        self.agent.train(False)

        traj1 = torch.tensor(traj1).to(self.device).float()
        traj2 = torch.tensor(traj2).to(self.device).float()
        traj1 = traj1.permute(0, 3, 1, 2)
        traj2 = traj2.permute(0, 3, 1, 2)
        
        # encode task emb
        task_emb1 = torch.tensor(task_emb1).to(self.device).float()
        lang_features1 = self.agent.language_projector(task_emb1[None]).repeat(traj1.shape[0], 1)
        task_emb2 = torch.tensor(task_emb2).to(self.device).float()
        lang_features2 = self.agent.language_projector(task_emb2[None]).repeat(traj2.shape[0], 1)
        
        # encoder traj
        traj1 = self.agent.encoder(traj1, lang=lang_features1)
        traj2 = self.agent.encoder(traj2, lang=lang_features2)

        # pad start of shorter traj with zeros if not OT based distance
        if not metric.startswith('sinkhorn'):
            if traj1.shape[0] < traj2.shape[0]:
                pad = torch.zeros(traj2.shape[0] - traj1.shape[0], traj1.shape[1]).to(self.device)
                traj1 = torch.cat([pad, traj1])
            elif traj2.shape[0] < traj1.shape[0]:
                pad = torch.zeros(traj1.shape[0] - traj2.shape[0], traj2.shape[1]).to(self.device)
                traj2 = torch.cat([pad, traj2])

        # Compute distance be tween traj1 and traj2
        if metric == 'cosine':
            dist = 1. - torch.nn.functional.cosine_similarity(traj1, traj2, dim=1)
        elif metric == 'euclidean':
            dist = torch.nn.functional.pairwise_distance(traj1, traj2, p=2)
        elif metric == 'sinkhorn_cosine':
            cost_matrix = cosine_distance(traj1, traj2)
            transport_plan = optimal_transport_plan(
                traj1, traj2, cost_matrix, method='sinkhorn',
                niter=100).float()  # Getting optimal coupling
            dist = torch.diag(torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()
        elif metric == 'sinkhorn_euclidean':
            cost_matrix = euclidean_distance(traj1, traj2)
            transport_plan = optimal_transport_plan(
                traj1, traj2, cost_matrix, method='sinkhorn',
                niter=100).float()  # Getting optimal coupling
            dist = torch.diag(torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()
        else:
            raise NotImplementedError()
        
        return dist
    
    def compare(self):
        self.agent.train(False)
        task_idx1, task_idx2 = 0, 0
        traj_idx1, traj_idx2 = 0, 5
        metric = 'cosine' # cosine, euclidean, sinkhorn_cosine, sinkhorn_euclidean
        
        traj1 = self.expert_replay_loader.dataset._episodes[task_idx1][traj_idx1]['observation']['pixels5']
        traj2 = self.expert_replay_loader.dataset._episodes[task_idx2][traj_idx2]['observation']['pixels5']
        task_emb1 = self.expert_replay_loader.dataset._episodes[task_idx1][traj_idx1]['task_emb']
        task_emb2 = self.expert_replay_loader.dataset._episodes[task_idx2][traj_idx2]['task_emb']

        dist = self.compute_distance(traj1, traj2, task_emb1, task_emb2, metric)
        print(dist)
        import ipdb; ipdb.set_trace()
        print("Hello")
        

    def eval(self):
        self.agent.train(False)
        episode_rewards = []
        successes = []
        for env_idx in range(self.envs_till_idx):
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []

            while eval_until_episode(episode):
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0

                # prompt
                if self.cfg.prompt != None and self.cfg.prompt != "intermediate_goal":
                    prompt = self.expert_replay_loader.dataset.sample_test(env_idx)
                else:
                    prompt = None

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                # plot obs with cv2
                while not time_step.last():
                    if self.cfg.prompt == "intermediate_goal":
                        prompt = self.expert_replay_loader.dataset.sample_test(
                            env_idx, step
                        )
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            prompt,
                            self.expert_replay_loader.dataset.stats,
                            step,
                            self.global_step,
                            eval_mode=True,
                        )
                    time_step = self.env[env_idx].step(action)
                    self.video_recorder.record(self.env[env_idx])
                    total_reward += time_step.reward
                    step += 1

                    if self.cfg.suite.name == "calvin" and time_step.reward == 1:
                        self.agent.buffer_reset()

                episode += 1
                success.append(time_step.observation["goal_achieved"])
            self.video_recorder.save(f"{self.global_frame}_env{env_idx}.mp4")
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - self.envs_till_idx):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[: self.envs_till_idx]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f)
            agent_payload["vqvae"] = payload
        self.agent.load_snapshot(agent_payload, eval=True)


@hydra.main(config_path="cfgs", config_name="config_eval")
def main(cfg):
    from compare_traj import WorkspaceIL as W

    root_dir = Path.cwd()
    workspace = W(cfg)

    # Load weights
    snapshots = {}
    # bc
    bc_snapshot = Path(cfg.bc_weight)
    if not bc_snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
    print(f"loading bc weight: {bc_snapshot}")
    snapshots["bc"] = bc_snapshot
    workspace.load_snapshot(snapshots)

    # workspace.eval()
    workspace.compare()


if __name__ == "__main__":
    main()
