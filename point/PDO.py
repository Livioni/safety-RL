# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softplus
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="SafeRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="point-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--penalty-lr", type=float, default=5e-2,
        help="the learning rate of the penalty optimizer")
    parser.add_argument("--xlambda", type=float, default=1.0,
        help="the initial lambda")
    parser.add_argument("--vf-lr", type=float, default=1e-3,
        help="the learning rate of the value function optimizer")
    parser.add_argument("--cost-limit", type=float, default=5,
        help="the limit of cost per episode")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=3000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.97,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=20,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.012,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.hidden = 256
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),  self.hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden, self.hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden, self.hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden, envs.single_action_space.shape[0]), std=0.01),
        )
        self.coster = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden, self.hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden, 1), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1,envs.single_action_space.shape[0]))


    def get_value(self, x):
        return self.critic(x)

    def get_cvalue(self,x):
        return self.coster(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), self.coster(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    penalty_param = torch.tensor(args.xlambda,requires_grad=True).float()
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.vf_lr, eps=1e-5)
    cost_optimizer = optim.Adam(agent.coster.parameters(), lr=args.vf_lr, eps=1e-5)
    penalty_optimizer =optim.Adam([penalty_param], lr=args.penalty_lr)#惩罚系数优化器

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cvalues = torch.zeros((args.num_steps, args.num_envs)).to(device) 

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            actor_optimizer.param_groups[0]["lr"] = lrnow

        count = 0
        reward_pool = []
        length_pool = []
        cost_pool = []
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, cvalue = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                cvalues[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            cost_array = np.zeros(args.num_envs,dtype=np.float32)
            for i in range(args.num_envs):
                cost_array[i] = info[i]['cost']
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            costs[step] = torch.tensor(cost_array).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    count += 1
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    reward_pool.append(item["episode"]["r"])
                    length_pool.append(item["episode"]["l"])
                    cost_pool.append(item["episode"]["c"])
                    if count == 10:
                        writer.add_scalar("charts/episodic_return", np.mean(reward_pool), global_step)
                        writer.add_scalar("charts/episodic_length", np.mean(length_pool), global_step)
                        writer.add_scalar("charts/episodic_cost", np.mean(cost_pool), global_step)
                        count = 0
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
            next_cvalue = agent.get_cvalue(next_obs).reshape(1, -1)
            c_advantages = torch.zeros_like(costs).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextcvalues = next_cvalue
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextcvalues = cvalues[t + 1]
                delta = costs[t] + args.gamma * nextcvalues * nextnonterminal - cvalues[t]
                c_advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            creturns = c_advantages + cvalues

            #calculate average ep cost
            episode_cost = []
            accumulate_cost = torch.zeros([args.num_envs],dtype=torch.float32)
            for t in range(args.num_steps):
                if torch.eq(dones[t],torch.zeros([args.num_envs],dtype=torch.float32)).all().item() is True :
                    accumulate_cost += costs[t]
                else: 
                    indx = torch.where(dones[t]==1)
                    for add in indx[0]:
                        episode_cost.append(accumulate_cost[add.item()].item())
                        accumulate_cost[add.item()] = 0
            if episode_cost == []:
                average_ep_cost = 0
            else:
                average_ep_cost = np.mean(episode_cost)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_costs = costs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_cadvantages = c_advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_creturns = creturns.reshape(-1)
        b_values = values.reshape(-1)
        b_cvalues = cvalues.reshape(-1)

        # Optimizing the lambda
        cost_devitation = average_ep_cost - args.cost_limit
        #update lambda
        loss_penalty = - penalty_param * cost_devitation
        penalty_optimizer.zero_grad()
        loss_penalty.backward()
        penalty_optimizer.step()

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, newcvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_cadvantages = b_cadvantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    mb_cadvantages = (mb_cadvantages - mb_cadvantages.mean()) / (mb_cadvantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.min(pg_loss1, pg_loss2).mean()
                
                cpg_loss = ratio * mb_cadvantages
                cpg_loss = cpg_loss.mean()

                p = softplus(penalty_param)
                penalty_item = p.item()

                entropy_loss = entropy.mean()
                # Create policy objective function, including entropy regularization
                objective = pg_loss + args.ent_coef * entropy_loss

                # Possibly include cpg_loss in objective
                objective -= penalty_item * cpg_loss
                objective = -objective/(1+penalty_item)

                actor_optimizer.zero_grad()
                objective.backward()
                actor_optimizer.step()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                newcvalue = newcvalue.view(-1)
                if args.clip_vloss:
                    cv_loss_unclipped = (newcvalue - b_creturns[mb_inds]) ** 2
                    cv_clipped = b_cvalues[mb_inds] + torch.clamp(
                        newcvalue - b_cvalues[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    cv_loss_clipped = (cv_clipped - b_creturns[mb_inds]) ** 2
                    cv_loss_max = torch.max(cv_loss_unclipped, cv_loss_clipped)
                    cv_loss = 0.5 * cv_loss_max.mean()
                else:
                    cv_loss = 0.5 * ((newcvalue - b_creturns[mb_inds]) ** 2).mean()                    

                critic_optimizer.zero_grad()
                v_loss.backward()
                critic_optimizer.step()

                cost_optimizer.zero_grad()
                cv_loss.backward()
                cost_optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    print(f"Early Stopping at epoch {epoch} due to reaching max KL")
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/cost_value_loss", cv_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
