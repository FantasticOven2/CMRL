import os
import argparse
from copy import deepcopy
from tqdm import trange
import pandas as pd

from mushroom_rl.algorithms.actor_critic import PPO, TRPO, DDPG, TD3, SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.policy import GaussianTorchPolicy, OrnsteinUhlenbeckPolicy, ClippedGaussianPolicy
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from atacom.environments.circular_motion import CircleEnvAtacom, CircleEnvTerminated, CircleEnvErrorCorrection
from network import *

def experiment(mdp, agent, seed, results_dir, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
                quiet, render, **kwargs):

    build_params = kwargs['build_params']
    logger = Logger(results_dir=results_dir, seed=seed, log_name='exp')

    logger.strong_line()
    logger.info('Experiment Algorithm: ' + type(agent).__name__)
    logger.info('Environment: ' + type(mdp).__name__ + " seed: " + str(seed))

    best_agent = deepcopy(agent)

    core = Core(agent, mdp)

    eval_params = dict(
        n_episodes = n_episodes_test,
        render = render,
        quiet = quiet
    )

    J, R, E, c_avg, c_max, c_dq_max = compute_metrics(core, eval_params, build_params)
    best_J, best_R, best_E, best_c_avg, best_c_max, best_c_dq_max = J, R, E, c_avg, c_max, c_dq_max

    logger.epoch_info(0, J=J, R=R, E=E, c_avg=c_avg, c_max=c_max, c_dq_max=c_dq_max)
    logger.weak_line()
    logger.log_numpy(J=J, R=R, E=E, c_avg=c_avg, c_max=c_max, c_dq_max=c_dq_max)

    for it in trange(n_epochs, leave=False, disable=quiet):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=quiet, render=render)
        J, R, E, c_avg, c_max, c_dq_max = compute_metrics(core, eval_params, build_params)

        logger.epoch_info(it + 1, J=J, R=R, E=E, c_avg=c_avg, c_max=c_max, c_dq_max=c_dq_max)
        logger.log_numpy(J=J, R=R, E=E, c_avg=c_avg, c_max=c_max, c_dq_max=c_dq_max)

        if J > best_J:
            best_J = J
            best_R = R
            best_E = E
            best_c_avg = c_avg
            best_c_max = c_max
            best_c_dq_max = c_dq_max
            best_agent = deepcopy(agent)

        if it % 10 == 0:
            logger.log_agent(agent, epoch=it)
    
    logger.info("Best result | J: {}, R: {}, E: {}, c_avg: {}, c_max: {}, c_dq_max{}.".format(best_J, best_R, best_E, 
                                                                                                best_c_avg, best_c_max,
                                                                                                best_c_dq_max))

    logger.strong_line()
    logger.log_agent(best_agent)
    best_res = {"best_J": best_J, "best_R": best_R, "best_E": best_E,
                "best_c_avg": best_c_avg, "best_c_max": best_c_max, "best_c_dq_max": best_c_dq_max}
    best_res = pd.DataFrame.from_dict(best_res, orient='index')
    best_res.to_csv(os.path.join(logger.path, "best_result.csv"))


def compute_metrics(core, eval_params, build_params):
    dataset = core.evaluate(**eval_params)
    c_avg, c_max, c_dq_max = core.mdp.get_constraints_logs()
    J = np.mean(compute_J(dataset, core.mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = None
    if build_params['compute_policy_entropy']:
        if build_params['compute_entropy_with_states']:
            E = core.agent.policy.entropy(parse_dataset(dataset)[0])
        else:
            E = core.agent.policy.entropy()
    return J, R, E, c_avg, c_max, c_dq_max


def build_env(env, horizon, gamma, random_init, **kwargs):
    mdp = CircleEnvAtacom(horizon=horizon, gamma=gamma, random_init=random_init, Kc=100)


def build_agent(alg, mdp_info, **kwargs):
    alg = alg.upper()
    if alg == 'PPO':
        agent, build_params = build_agent_PPO(mdp_info, **kwargs)
    else:
        raise NotImplementedError
    reutrn agent, build_params


def build_agent_PPO(mdp_info, actor_lr, critic_lr, n_features, batch_size, eps_ppo, lam, ent_coeff, **kwargs):
    policy_params = dict(
        std_0 = 0.5,
        n_features = n_features,
        use_cuda = torch.cuda.is_available()
    )
    policy = GaussianTorchPolicy(PPONetwork, 
                                    mdp_info.observation_space.shape,
                                    mdp_info.action_space.shape,
                                    **policy_params)

    critic_params = dict(network=PPONetwork,
                            optimizer={'class': optim.Adam,
                                        'params': {'lr': critic_lr}},
                            loss=F.mse_loss,
                            n_features=n_features,
                            batch_size=batch_size,
                            input_shape=mdp_info.observation_space.shape,
                            output_shape=(1,))
    
    ppo_param = dict(actor_optimizer={'class': optim.Adam,
                                        'params': {'lr': actor_lr}},
                                        n_epochs_policy=4,
                                        batch_size=batch_size,
                                        eps_ppo=eps_ppo,
                                        lam=lam,
                                        ent_coeff=ent_coeff,
                                        critic_params=critic_params)

    build_params = dict(compute_entropy_with_states=False,
                        compute_policy_entropy=True)
    
    return PPO(mdp_info, policy, **ppo_params), build_params

                                
def default_params():
    defaults = dict(env='A', alg='TRPO', seed=1,
                    horizon=500, gamma=0.99, random_init=False, quiet=False, termination_tol=0.4, render=False,
                    results_dir='../logs/circular_motion')
    training_params = dict(n_epochs=50, n_steps=5000, n_steps_per_fit=1000, n_episodes_test=25)

    network_params = dcit(actor_lr=3e-4, critic_lr=3e-4, n_features=[32, 32], batch_size=64)

    trpo_ppo_params = dict([lam=0.95, ent_coeff=5e-5])
    ppo_params = dict(eps_ppo=0.1)
    trpo_params = dcit(max_kl=1e-2, n_epochs_line_search=10, n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10,
                        critic_fit_params=None)
    
    ddpg_td3_sac_params = dict(initial_replay_size=5000, max_replay_size=200000, tau=1e-3)
    td3_params = dict(sigma=1.0)

    sac_params = dict(warmup_transitions=10000, lr_alpha=3e-3, target_entropy=-6)

    defaults.update(training_params)
    defaults.update(network_params)
    defaults.update(trpo_ppo_params)
    defaults.update(ppo_params)

    ### Things left for non PPO alg ###
    return defaults

def parse_args():
    parser = argparse.ArgumentParser()

    arg_test = parser.add_argument_group('Experiment')
    arg_test.add_argument('--env', choices=['A', 'T', 'E'], help="Environment argument ['A', 'T', 'E']: "
                                                                 "'A' for ATACOM, "
                                                                 "'T' for TerminatedCircle, "
                                                                 "'E' for ErrorCorrection")
    arg_test.add_argument('--alg', choices=['TRPO', 'trpo', 'PPO', 'ppo', 'DDPG', 'ddpg', 'TD3', 'td3', 'SAC', 'sac'])

    arg_test.add_argument('--horizon', type=int)
    arg_test.add_argument('--gamma', type=float)
    arg_test.add_argument('--random-init', action="store_true")
    arg_test.add_argument('--termination-tol', type=float)
    arg_test.add_argument('--quiet', action="store_true")
    arg_test.add_argument('--render', action="store_true")

    # training parameter
    arg_test.add_argument('--n-epochs', type=int)
    arg_test.add_argument('--n-steps', type=int)
    arg_test.add_argument('--n-steps-per-fit', type=int)
    arg_test.add_argument('--n-episodes-test', type=int)

    # network parameter
    arg_test.add_argument('--actor-lr', type=float)
    arg_test.add_argument('--critic-lr', type=float)
    arg_test.add_argument('--n-features', nargs='+')
    arg_test.add_argument('--batch-size', type=int)

    # TRPO PPO parameter
    arg_test.add_argument('--lam', type=float)
    arg_test.add_argument('--ent-coeff', type=float)

    # PPO parameters
    arg_test.add_argument('--eps-ppo', type=float)

    # TRPO parameters
    arg_test.add_argument('--max-kl', type=float)
    arg_test.add_argument('--n-epochs-line-search', type=int)
    arg_test.add_argument('--n-epochs-cg', type=int)
    arg_test.add_argument('--cg-damping', type=float)
    arg_test.add_argument('--cg-residual-tol', type=float)

    # DDPG TD3 parameters
    arg_test.add_argument('--initial-replay-size', type=int)
    arg_test.add_argument('--max-replay-size', type=int)
    arg_test.add_argument('--tau', type=float)

    # TD3 parameters
    arg_test.add_argument('--sigma', type=float)

    # SAC parameters
    arg_test.add_argument('--warmup-transitions', type=int)
    arg_test.add_argument('--lr-alpha', type=float)
    arg_test.add_argument('--target-entropy', type=float)

    arg_default = parser.add_argument_group('Default')
    arg_default.add_argument('--seed', type=int)
    arg_default.add_argument('--results-dir', type=str)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args_ = parse_args()
    env_ = build_envs(**args_)
    agent_, build_params_ = build_agent(mdp_info=env_.info, **args_)
    experiment(env_, agent_, build_params=build_params_, **args_)