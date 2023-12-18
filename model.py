import torch
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GENConv, GraphConv, JumpingKnowledge, GATv2Conv, GINConv
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout


class Net(torch.nn.Module):
    def __init__(self, num_layers, num_node_features, num_classes):
        super(Net, self).__init__()

        self.num_layers = num_layers

        # Create a list to hold SAGEConv layers
        self.conv_layers = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(GENConv(in_channels=num_node_features, out_channels=512,
                                                aggr='power', p=1, learn_p=True, msg_norm=True, learn_msg_scale=True,
                                                norm='layer', num_layers=1))
            else:
                self.conv_layers.append(GENConv(in_channels=512, out_channels=512, aggr='power', p=1,
                                                learn_p=True, msg_norm=True, learn_msg_scale=True,
                                                norm='layer', num_layers=1))

        self.batch_norm_layers = torch.nn.ModuleList([BatchNorm1d(512) for _ in range(num_layers)])

        self.fc1 = Linear(2 * 512, 512)
        self.batch_norm4 = BatchNorm1d(512)
        self.dropout = Dropout(0.65)
        self.fc2 = Linear(512, num_classes)

        # Create JumpingKnowledge module
        self.jk = JumpingKnowledge(mode='max')

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        node_representations = []

        # Forward pass through GATv2Conv layers
        for i in range(self.num_layers):
            # print('layer {}'.format(i))
            # print('shape of x before conv', x.shape, batch.shape)
            x = self.conv_layers[i](x, edge_index)
            # print('shape of x after conv', x.shape, batch.shape)
            x = self.batch_norm_layers[i](x)
            x = F.relu(x)

            # Collect intermediate node representations
            node_representations.append(x)

        # Apply JumpingKnowledge aggregation to node representations
        # print('shape of x before jk', x.shape)
        x = self.jk(node_representations)

        # Concatenate global max-pooling and global average-pooling representations
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # Continue with the rest of the layers as before
        x = self.fc1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (Linear, GENConv, BatchNorm1d)):
                module.reset_parameters()

