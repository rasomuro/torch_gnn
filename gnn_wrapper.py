import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import Dataset
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from utils import Accuracy
from torch.utils.tensorboard import SummaryWriter
import torchvision
from utils import matplotlib_imshow
import utils
from pygnn import GNN

from typing import Optional, Literal


class GNNWrapper:
    class Config:
        device: Optional[torch.device]
        use_cuda: bool
        dataset_path: Optional[str]
        log_interval: int
        tensorboard: bool
        task_type: Literal['binary','multiclass','semisupervised','multilabel']
        lrw: float
        epochs: int
        convergence_threshold: float
        max_iterations: int
        n_nodes: int
        state_dim: int
        label_dim: int
        edge_label_dim: int
        output_dim: int
        graph_based: bool
        activation: nn.Module
        state_transition_hidden_dims: list[int]
        output_function_hidden_dims: list[int]

        def __init__(self):
            self.device = None
            self.use_cuda = False
            self.dataset_path = None
            self.log_interval = 10
            self.tensorboard = False

            # hyperparams
            self.lrw = .05
            self.epochs = 50
            self.convergence_threshold = .01
            self.max_iterations = 50
            self.graph_based = False
            self.activation = torch.nn.Tanh()
            self.task_type = "multilabel"

    optimizer: torch.optim.Optimizer
    criterion: nn.CrossEntropyLoss # TODO: is there a loss superclass?
    tr_data: Dataset
    ts_data: Optional[Dataset]

    def __init__(self, config: Config):
        self.config = config

        # to be populated
        self.train_loader = None
        self.test_loader = None

        if self.config.tensorboard:
            self.writer = SummaryWriter('logs/tensorboard')
        self.first_flag_writer = True

    def __call__(self, tr_dset, ts_dset=None, state_net=None, out_net=None):
        # handle the dataset info
        self._data_loader(tr_dset, ts_dset)
        self.gnn = GNN(self.config, state_net, out_net).to(self.config.device)
        self._criterion()
        self._optimizer()
        self._accuracy()

    def _data_loader(self, tr_dset, ts_dset=None):  # handle dataset data and metadata
        self.tr_dset = tr_dset.to(self.config.device)
        self.ts_dset = None if ts_dset is None else ts_dset.to(self.config.device)
        self.config.label_dim = self.tr_dset.node_label_dim
        self.config.edge_label_dim = self.tr_dset.edge_label_dim
        self.config.n_nodes = self.tr_dset.num_nodes
        self.config.output_dim = self.tr_dset.num_classes

    def _optimizer(self):
        # for name, param in self.gnn.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # exit()
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=self.config.lrw)
        # self.optimizer = optim.SGD(self.gnn.parameters(), lr=self.config.lrw)

    def _criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type=self.config.task_type)
        self.ValidAccuracy = Accuracy(type=self.config.task_type)
        self.TestAccuracy = Accuracy(type=self.config.task_type)

    def train_step(self, epoch):
        self.gnn.train()
        data = self.tr_dset
        self.optimizer.zero_grad()
        self.TrainAccuracy.reset()
        # output computation
        if self.config.graph_based:
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, graph_agg=data.graph_node, edge_labels=data.edge_labels)
        else:
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, edge_labels=data.edge_labels)
        # loss computation - semisupervised
        loss = self.criterion(output, data.targets)

        loss.backward()

        self.optimizer.step()

        # # updating accuracy
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():  # Accuracy computation
            # accuracy_train = torch.mean(
            #     (torch.argmax(output[data.idx_train], dim=-1) == data.targets[data.idx_train]).float())
            self.TrainAccuracy.update(output, data.targets)
            accuracy_train = self.TrainAccuracy.compute()

            if epoch % self.config.log_interval == 0:
                print(
                    'Train Epoch: {} \t Mean Loss: {:.6f}\tAccuracy Full Batch: {:.6f} \t  Best Accuracy : {:.6f}  \t Iterations: {}'.format(
                        epoch, loss, accuracy_train, self.TrainAccuracy.get_best(), iterations))

                if self.config.tensorboard:
                    self.writer.add_scalar('Training Accuracy',
                                           accuracy_train,
                                           epoch)
                    self.writer.add_scalar('Training Loss',
                                           loss,
                                           epoch)
                    self.writer.add_scalar('Training Iterations',
                                           iterations,
                                           epoch)

                    for name, param in self.gnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
        # self.TrainAccuracy.reset()

    def predict(self, edges, agg_matrix, node_labels, *, graph_node=None, edge_labels=None):
        return self.gnn(edges, agg_matrix, node_labels, graph_agg=graph_node, edge_labels=edge_labels)

    def test_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.tr_dset if self.ts_dset is None else self.ts_dset
        self.TestAccuracy.reset()
        with torch.no_grad():
            if self.config.graph_based:
                output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, graph_agg=data.graph_node, edge_labels=data.edge_labels)
            else:
                output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, edge_labels=data.edge_labels)
            test_loss = self.criterion(output, data.targets)

            self.TestAccuracy.update(output, data.targets)
            acc_test = self.TestAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Test set: Average loss: {:.4f}, Accuracy:  ({:.2f}%) , Best Accuracy:  ({:.2f}%)'.format(
                    test_loss, 100*acc_test, 100*self.TestAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Test Accuracy',
                                           acc_test,
                                           epoch)
                    self.writer.add_scalar('Test Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Test Iterations',
                                           iterations,
                                           epoch)

    def valid_step(self, epoch): # TODO
        ####  TEST
        self.gnn.eval()
        data = self.tr_dset if self.ts_dset is None else self.ts_dset
        self.ValidAccuracy.reset()
        with torch.no_grad():
            if self.config.graph_based:
                output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, graph_agg=data.graph_node, edge_labels=data.edge_labels)
            else:
                output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, edge_labels=data.edge_labels)
            test_loss = self.criterion(output, data.targets)

            self.ValidAccuracy.update(output, data.targets)
            acc_valid = self.ValidAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Valid set: Average loss: {:.4f}, Accuracy:  ({:.2f}%) , Best Accuracy:  ({:.2f}%)'.format(
                    test_loss, 100*acc_valid, 100*self.ValidAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Valid Accuracy',
                                           acc_valid,
                                           epoch)
                    self.writer.add_scalar('Valid Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Valid Iterations',
                                           iterations,
                                           epoch)


class SemiSupGNNWrapper(GNNWrapper):
    class Config(GNNWrapper.Config):
        def __init__(self):
            super().__init__()
            self.task_type = "semisupervised"

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type="semisupervised")
        self.ValidAccuracy = Accuracy(type="semisupervised")
        self.TestAccuracy = Accuracy(type="semisupervised")

    def train_step(self, epoch):
        self.gnn.train()
        data = self.tr_dset
        self.optimizer.zero_grad()
        self.TrainAccuracy.reset()
        # output computation
        if self.config.graph_based:
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, data.graph_node, edge_labels=data.edge_labels)
        else:
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, edge_labels=data.edge_labels)
        # loss computation - semisupervised
        loss = self.criterion(output[data.idx_train], data.targets[data.idx_train])

        loss.backward()

        # with torch.no_grad():
        #     for name, param in self.gnn.named_parameters():
        #         if "state_transition_function" in name:
        #             #self.writer.add_histogram("gradient " + name, param.grad, epoch)
        #             param.grad = 0*  param.grad



        self.optimizer.step()

        # # updating accuracy
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():  # Accuracy computation
            # accuracy_train = torch.mean(
            #     (torch.argmax(output[data.idx_train], dim=-1) == data.targets[data.idx_train]).float())
            self.TrainAccuracy.update(output, data.targets, idx=data.idx_train)
            accuracy_train = self.TrainAccuracy.compute()

            if epoch % self.config.log_interval == 0:
                print(
                    'Train Epoch: {} \t Mean Loss: {:.6f}\tAccuracy Full Batch: {:.6f} \t  Best Accuracy : {:.6f}  \t Iterations: {}'.format(
                        epoch, loss, accuracy_train, self.TrainAccuracy.get_best(), iterations))

                if self.config.tensorboard:
                    self.writer.add_scalar('Training Accuracy',
                                           accuracy_train,
                                           epoch)
                    self.writer.add_scalar('Training Loss',
                                           loss,
                                           epoch)
                    self.writer.add_scalar('Training Iterations',
                                           iterations,
                                           epoch)
                    for name, param in self.gnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
                        self.writer.add_histogram("gradient " + name, param.grad, epoch)
        # self.TrainAccuracy.reset()
        return output  # used for plotting

    def test_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.tr_dset if self.ts_dset is None else self.ts_dset
        self.TestAccuracy.reset()
        with torch.no_grad():
            if self.config.graph_based:
                output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, data.graph_node, edge_labels=data.edge_labels)
            else:
                output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, edge_labels=data.edge_labels)
            test_loss = self.criterion(output[data.idx_test], data.targets[data.idx_test])

            self.TestAccuracy.update(output, data.targets, idx=data.idx_test)
            acc_test = self.TestAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Test set: Average loss: {:.4f}, Accuracy:  ({:.2f}%) , Best Accuracy:  ({:.2f}%)'.format(
                    test_loss, 100*acc_test, 100*self.TestAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Test Accuracy',
                                           acc_test,
                                           epoch)
                    self.writer.add_scalar('Test Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Test Iterations',
                                           iterations,
                                           epoch)

    def valid_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.tr_dset if self.ts_dset is None else self.ts_dset
        self.ValidAccuracy.reset()
        with torch.no_grad():
            if self.config.graph_based:
                output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, data.graph_node, edge_labels=data.edge_labels)
            else:
                output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels, edge_labels=data.edge_labels)
            test_loss = self.criterion(output[data.idx_valid], data.targets[data.idx_valid])

            self.ValidAccuracy.update(output, data.targets, idx=data.idx_valid)
            acc_valid = self.ValidAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Valid set: Average loss: {:.4f}, Accuracy:  ({:.2f}%) , Best Accuracy:  ({:.2f}%)'.format(
                    test_loss, 100*acc_valid, 100*self.ValidAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Valid Accuracy',
                                           acc_valid,
                                           epoch)
                    self.writer.add_scalar('Valid Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Valid Iterations',
                                           iterations,
                                           epoch)
