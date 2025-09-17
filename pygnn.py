import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
from net import MLP, StateTransition


class GNN(nn.Module):

    def __init__(self, config, state_net=None, out_net=None):
        super(GNN, self).__init__()

        self.config = config
        # hyperparameters and general properties
        self.convergence_threshold = config.convergence_threshold
        self.max_iterations = config.max_iterations
        self.state_dim = config.state_dim
        self.label_dim = config.label_dim
        self.output_dim = config.output_dim
        self.state_transition_hidden_dims = config.state_transition_hidden_dims
        self.output_function_hidden_dims = config.output_function_hidden_dims

        # state and output transition functions
        if state_net is None:
            self.state_transition_function = StateTransition(self.state_dim, self.label_dim,
                                                             mlp_hidden_dim=self.state_transition_hidden_dims,
                                                             activation_function=config.activation)
        else:
            self.state_transition_function = state_net
        if out_net is None:
            self.output_function = MLP(self.state_dim, self.output_function_hidden_dims, self.output_dim)
        else:
            self.output_function = out_net

        self.graph_based = self.config.graph_based

    def reset_parameters(self):
        self.state_transition_function.mlp.init()
        self.output_function.init()

    def forward(self,
                edges,
                agg_matrix,
                node_labels,
                node_states=None,
                graph_agg=None
                ):
        n_iterations = 0
        # convergence loop
        # state initialization
        if node_states is None:
            node_states = torch.zeros(*[node_labels.shape[0], self.state_dim]).to(self.config.device)  # (n,d_n)

        while n_iterations < self.max_iterations:
            new_state = self.state_transition_function(node_states, node_labels, edges, agg_matrix)
            n_iterations += 1
            # convergence condition
            with torch.no_grad():
                distance = torch.norm(input=new_state - node_states,
                                      dim=1)  # checked, they are the same (in cuda, some bug)

                check_min = distance < self.convergence_threshold
            node_states.copy_(new_state)

            if check_min.all():
                break

        output = self.output_function(torch.matmul(graph_agg, node_states) if self.graph_based else node_states)

        return output, n_iterations
