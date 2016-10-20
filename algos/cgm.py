from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import CGMBatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
import copy
import numpy as np
class CGM(CGMBatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            agentNum=1,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = []
        self.agentNum=agentNum
        for i in xrange(agentNum):
            self.optimizer.append(copy.deepcopy(optimizer))
        self.step_size = step_size
        super(CGM, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        for i in xrange(self.agentNum):
            optimizer = self.optimizer[i]
            policy = self.policy._components[i]
            is_recurrent = int(policy.recurrent)
            obs_var = self.env.observation_space._components[i].new_tensor_variable(
                'obs',
                extra_dims=1 + is_recurrent,
            )
            action_var = self.env.action_space._components[i].new_tensor_variable(
                'action',
                extra_dims=1 + is_recurrent,
            )
            advantage_var = ext.new_tensor(
                'advantage',
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            )
            dist = policy.distribution
            old_dist_info_vars = {
                k: ext.new_tensor(
                    'old_%s' % k,
                    ndim=2 + is_recurrent,
                    dtype=theano.config.floatX
                ) for k in dist.dist_info_keys
                }
            old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

            state_info_vars = {
                k: ext.new_tensor(
                    k,
                    ndim=2 + is_recurrent,
                    dtype=theano.config.floatX
                ) for k in policy.state_info_keys
            }
            state_info_vars_list = [state_info_vars[k] for k in policy.state_info_keys]

            if is_recurrent:
                valid_var = TT.matrix('valid')
            else:
                valid_var = None

            dist_info_vars = policy.dist_info_sym(obs_var, state_info_vars)
            kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
            lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
            if is_recurrent:
                mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
                surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
            else:
                mean_kl = TT.mean(kl)
                surr_loss = - TT.mean(lr * advantage_var)

            input_list = [
                             obs_var,
                             action_var,
                             advantage_var,
                         ] + state_info_vars_list + old_dist_info_vars_list
            if is_recurrent:
                input_list.append(valid_var)

            optimizer.update_opt(
                loss=surr_loss,
                target=policy,
                leq_constraint=(mean_kl, self.step_size),
                inputs=input_list,
                constraint_name="mean_kl"
            )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        jointFeatures = ext.jointextract(
            samples_data, len(self.optimizer),
            "observations", "actions", "advantages"
        )
        for i in xrange(self.agentNum):
            if i == 19:
                temp = 1
            optimizer = self.optimizer[i]
            policy = self.policy._components[i]
            agent_infos = samples_data["agent_infos"]
            state_info_list = []
            for k in policy.state_info_keys:
                state_info_list.append(agent_infos[k][:,i])
            dist_info_list = []
            for k in policy.distribution.dist_info_keys:
                dist_info_list.append(agent_infos[k][:,i])
            all_input_values = jointFeatures[i] + tuple(state_info_list) + tuple(dist_info_list)
            if policy.recurrent:
                all_input_values += (samples_data["valids"],)
            loss_before = optimizer.loss(all_input_values)
            optimizer.optimize(all_input_values)
            mean_kl = optimizer.constraint_val(all_input_values)
            loss_after = optimizer.loss(all_input_values)
            logger.record_tabular('LossBefore', loss_before)
            logger.record_tabular('LossAfter', loss_after)
            logger.record_tabular('MeanKL', mean_kl)
            logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
