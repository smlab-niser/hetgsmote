import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

# Content Aggregation Layer
class MAG_Content_Agg(nn.Module):
    def __init__(self, embed_dim, nux_node, dropout):
        super(MAG_Content_Agg, self).__init__()
        self.fc = nn.Linear(embed_dim * nux_node, embed_dim)
        self.dropout = dropout

    def forward(self, embed_list):
        x = torch.cat(embed_list, dim=-1)
        x = F.normalize(F.relu(self.fc(x)), p=2, dim=-1)
        # x = F.dropout(x, self.dropout, training=self.training)
        return x


class MAGNN_Agg(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(MAGNN_Agg, self).__init__()
        self.fc_s1s = nn.Linear(embed_dim, embed_dim)
        self.fc_s2s = nn.Linear(embed_dim, embed_dim)
        self.fc_s12s = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

        self.att_vec = nn.Parameter(torch.Tensor(3, embed_dim))
        nn.init.xavier_uniform_(self.att_vec)

    def forward(self, x_list, edge_index_list, x_node, edge_weight_list, edge_index_12):

        edge_index_s1 = edge_index_list[1]
        edge_index_s2 = edge_index_list[2]

        if edge_weight_list[0] is None: 
            edge_weight_list = [torch.ones(edge_index.size(1), device=x_node.device) for edge_index in edge_index_list]

        # s-1-s aggregation
        msg_1 = scatter_mean(x_node[edge_index_s1[0]] * edge_weight_list[1].view(-1, 1), edge_index_s1[1], dim=0, dim_size=x_list[1].size(0))
        net_msg1 = (msg_1 + x_list[1]) / 2
        s1s_agg = scatter_mean(net_msg1[edge_index_s1[1]], edge_index_s1[0], dim=0, dim_size=x_node.size(0))
        s1s_agg = F.relu(self.fc_s1s(s1s_agg))
        s1s_agg = F.dropout(s1s_agg, self.dropout, training=self.training)

        # s-2-s aggregation
        msg_2 = scatter_mean(x_node[edge_index_s2[0]] * edge_weight_list[2].view(-1, 1), edge_index_s2[1], dim=0, dim_size=x_list[2].size(0))
        net_msg2 = (msg_2 + x_list[2]) / 2
        s2s_agg = scatter_mean(net_msg2[edge_index_s2[1]], edge_index_s2[0], dim=0, dim_size=x_node.size(0))
        s2s_agg = F.relu(self.fc_s2s(s2s_agg))
        s2s_agg = F.dropout(s2s_agg, self.dropout, training=self.training)

        # s-1-2-s aggregation
        # s -> 1
        msg_1_s12s = scatter_mean(x_node[edge_index_s1[0]] * edge_weight_list[1].view(-1, 1), edge_index_s1[1], dim=0, dim_size=x_list[1].size(0))
        net_msg1_s12s = (msg_1_s12s+ x_list[1]) / 2
        # 1 -> 2
        msg_2_s12s = scatter_mean(net_msg1_s12s[edge_index_12[0]], edge_index_12[1], dim=0, dim_size=x_list[2].size(0))
        net_msg2_s12s = (msg_2_s12s+ x_list[2]) / 2
        # 2 -> 1
        s12s_agg = scatter_mean(net_msg2_s12s[edge_index_s2[1]] * edge_weight_list[2].view(-1, 1), edge_index_s2[0], dim=0, dim_size=x_node.size(0))
        s12s_agg = F.relu(self.fc_s12s(s12s_agg))
        s12s_agg = F.dropout(s12s_agg, self.dropout, training=self.training)

        # Stack all meta-path embeddings for inter-metapath attention
        all_metapath = torch.stack([s1s_agg, s2s_agg, s12s_agg], dim=1)  # [num_nodes, 3, embed_dim]

        # Attention score computation
        att_scores = (all_metapath * self.att_vec.unsqueeze(0)).sum(dim=2)  # [num_nodes, 3]
        att_weights = F.softmax(att_scores, dim=1).unsqueeze(-1)            # [num_nodes, 3, 1]

        h_final = torch.sum(all_metapath * att_weights, dim=1)              # [num_nodes, embed_dim]

        return h_final
    

class MAG_ConEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(MAG_ConEn, self).__init__()

        self.m_con = MAG_Content_Agg(embed_dim, 4, dropout)
        self.a_con = MAG_Content_Agg(embed_dim, 2, dropout)
        self.d_con = MAG_Content_Agg(embed_dim, 2, dropout)
        
        self.m_cont = torch.empty(0)
        self.a_cont = torch.empty(0)
        self.d_cont = torch.empty(0)
        
    def forward(self, data):
        
        m_embed_list = [data['m_text_embed'].x, data['m_net_embed'].x, data['m_a_net_embed'].x, data['m_d_net_embed'].x]
        a_embed_list = [data['a_net_embed'].x, data['a_text_embed'].x]
        d_embed_list = [data['d_net_embed'].x, data['d_text_embed'].x]
        
        self.m_cont = self.m_con(m_embed_list)
        self.a_cont = self.a_con(a_embed_list)
        self.d_cont = self.d_con(d_embed_list)
        
        return [self.m_cont, self.a_cont, self.d_cont]
            


class MAG_NetEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(MAG_NetEn, self).__init__()
        
        self.m_het = MAGNN_Agg(embed_dim, dropout)
        self.a_het = MAGNN_Agg(embed_dim, dropout)
        self.d_het = MAGNN_Agg(embed_dim, dropout)
        
    def forward(self, x_list, data):

        m_edges_list = [data['m', 'walk', 'm'].edge_index, data['m', 'walk', 'a'].edge_index, data['m', 'walk', 'd'].edge_index]
        a_edges_list = [data['a', 'walk', 'a'].edge_index, data['a', 'walk', 'm'].edge_index, data['a', 'walk', 'd'].edge_index]
        d_edges_list = [data['d', 'walk', 'd'].edge_index, data['d', 'walk', 'm'].edge_index, data['d', 'walk', 'a'].edge_index]
        edge_weight_list = [None, None, None]
        
        x_list[0] = self.m_het(x_list, m_edges_list, x_list[0], edge_weight_list, a_edges_list[2]) 
        x_list[1] = self.a_het([x_list[1], x_list[0], x_list[2]], a_edges_list, x_list[1], edge_weight_list, m_edges_list[2])
        x_list[2] = self.d_het([x_list[2], x_list[0], x_list[1]], d_edges_list, x_list[2], edge_weight_list, m_edges_list[1])
        
        return x_list


class MAG_classify(nn.Module):
    def __init__(self, embed_dim, nclass, dropout):
        super(MAG_classify, self).__init__()
        self.m_het = MAGNN_Agg(embed_dim, dropout)
        self.mlp = nn.Linear(embed_dim, nclass)    
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x_list, edge_list, edge_weight_list, edge_index_mid):
        x = self.m_het(x_list, edge_list, x_list[0], edge_weight_list, edge_index_mid)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


class EdgePredictor(nn.Module):
    def __init__(self, nembed, dropout=0.1):
        super(EdgePredictor, self).__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(nembed, nembed)
        self.lin2 = nn.Linear(nembed, nembed)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.lin1.weight,std=0.05)
        nn.init.normal_(self.lin2.weight,std=0.05)

    def forward(self, node_embed1, node_embed2):
        
        combine1 = self.lin1(node_embed1)
        combine2 = self.lin2(node_embed2)
        result = torch.mm(combine1, combine2.transpose(-1, -2))

        adj_out = torch.sigmoid(result)         # Apply sigmoid along dim=1 (rows)
        return adj_out
    