import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def swish(x):
    return x * torch.sigmoid(x)

class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 kernel_size,
                 dropout_rate,
                 time_emb_dim=None,
                 causal_conv=True):
        super(SingleStageModel, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2 ** i,
                                     num_f_maps,
                                     causal_conv=causal_conv,
                                     kernel_size=kernel_size,
                                     dropout_rate=dropout_rate))
            for i in range(num_layers)
        ])

    def forward(self, x, time_emb=None, feature_layer_indices=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:, :, None]

        if feature_layer_indices is None:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            out = []
            for l_id, layer in enumerate(self.layers):
                x = layer(x)
                if l_id in feature_layer_indices:
                    out.append(x)

            if len(out) > 0:
                out = torch.cat(out, 1)
            else:
                out = None

            return x, out


class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 d_model,
                 dropout_rate,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(d_model,
                                          d_model,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(d_model,
                                          d_model,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * (self.kernel_size - 1))]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)

class MixedConvAttModule(nn.Module):  # for encoder
    def __init__(self, num_layers, num_f_maps, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModule, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayer(num_f_maps, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  # 2 ** i

    def forward(self, x, time_emb=None, feature_layer_indices=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:, :, None]

        if feature_layer_indices is None:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            out = []
            for l_id, layer in enumerate(self.layers):
                x = layer(x)
                if l_id in feature_layer_indices:
                    out.append(x)

            if len(out) > 0:
                out = torch.cat(out, 1)
            else:
                out = None

            return x, out


class MixedConvAttentionLayer(nn.Module):

    def __init__(self, d_model, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayer, self).__init__()

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation

        assert (self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)

        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None

    def get_attn_indices(self, l, device):

        attn_indices = []

        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)
            # 3  3  3 ...                             (k=3, //1)
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)

            indices = [i + self.padding for i in range(s, e, step)]

            attn_indices.append(indices)

        attn_indices = np.array(attn_indices)

        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)

    def attention(self, x):

        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)

        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2], :], (-1,))

        x_q = self.att_linear_q(x)
        x_k = self.att_linear_k(x)
        x_v = self.att_linear_v(x)

        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)

        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1]))

        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)

        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2], :] >= self.padding,
            self.attn_indices[:x.shape[2], :] < att.shape[1] + self.padding
        )  # 1 keep, 0 mask

        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2)
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)

        return r

    def forward(self, x):

        x_drop = self.dropout(x)
        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop)
        out = self.ffn_block(self.norm(out1 + out2))

        return x + out


class MixedConvAttModuleV2(nn.Module):  # for decoder
    def __init__(self, num_layers, num_f_maps, input_dim_cross, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModuleV2, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayerV2(num_f_maps, input_dim_cross, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  # 2 ** i

    def forward(self, x, x_cross, time_emb=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:, :, None]

        for layer in self.layers:
            x = layer(x, x_cross)

        return x


class MixedConvAttentionLayerV2(nn.Module):

    def __init__(self, d_model, d_cross, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayerV2, self).__init__()

        self.d_model = d_model
        self.d_cross = d_cross
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation

        assert (self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)

        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None

    def get_attn_indices(self, l, device):

        attn_indices = []

        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)
            # 3  3  3 ...                             (k=3, //1)
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)

            indices = [i + self.padding for i in range(s, e, step)]

            attn_indices.append(indices)

        attn_indices = np.array(attn_indices)

        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)

    def attention(self, x, x_cross):

        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)

        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2], :], (-1,))

        x_q = self.att_linear_q(torch.cat([x, x_cross], 1))
        x_k = self.att_linear_k(torch.cat([x, x_cross], 1))
        x_v = self.att_linear_v(x)

        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)

        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1]))

        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)

        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2], :] >= self.padding,
            self.attn_indices[:x.shape[2], :] < att.shape[1] + self.padding
        )  # 1 keep, 0 mask

        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2)
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)

        return r

    def forward(self, x, x_cross):

        x_drop = self.dropout(x)
        x_cross_drop = self.dropout(x_cross)

        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop, x_cross_drop)

        out = self.ffn_block(self.norm(out1 + out2))

        return x + out