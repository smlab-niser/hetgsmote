import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class HAN_Content_Agg(nn.Module):
    def __init__(self, embed_dim, num_embed, dropout):
        super(HAN_Content_Agg, self).__init__()
        self.fc = nn.Linear(embed_dim * num_embed, embed_dim)
        self.dropout = dropout

    def forward(self, embed_list):
        # Concatenate the embeddings
        x = torch.cat(embed_list, dim=-1)
        x = F.normalize(F.relu(self.fc(x)), p=2, dim = -1)
        # x = F.dropout(x, self.dropout, training=self.training) 
        return x

class HAN_NeighAgg(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super(HAN_NeighAgg, self).__init__()
        self.gat_s_to_t = GATConv(embed_dim, embed_dim // heads, heads=heads, dropout=dropout)
        self.dropout = dropout

    def forward(self, x_s, x_t, edge_index, edge_weight=None):

        # Zero-pad x_s if smaller than x_t
        if x_s.size(0) < x_t.size(0):
            padding = torch.zeros((x_t.size(0) - x_s.size(0), x_s.size(1)), device=x_s.device)
            x_s = torch.cat([x_s, padding], dim=0)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x_s.device)
        
        # Aggregating s-type neighbors for t-type nodes
        t_agg = self.gat_s_to_t((x_s, x_t), edge_index, edge_attr=edge_weight)
        t_agg = F.relu(t_agg)
        # t_agg = F.dropout(t_agg, self.dropout, training=self.training)
        
        return t_agg
    

class HAN_Agg(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super(HAN_Agg, self).__init__()   
        self.g_aggregate = HAN_NeighAgg(embed_dim, heads, dropout)
        self.d_aggregate = HAN_NeighAgg(embed_dim, heads, dropout)
        self.c_aggregate = HAN_NeighAgg(embed_dim, heads, dropout)
        self.s_aggregate = HAN_NeighAgg(embed_dim, heads, dropout)

        self.u = nn.Parameter(torch.randn(2 * embed_dim, 1))
        nn.init.normal_(self.u, mean=0, std=1)  # Normalize self.u
        self.linear = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, x_list, edges_list, x_node, edge_weight_list):
        g_aggr = self.g_aggregate(x_list[0], x_node, edges_list[0][[1, 0], :], edge_weight_list[0])
        d_aggr = self.d_aggregate(x_list[1], x_node, edges_list[1][[1, 0], :], edge_weight_list[1])
        c_aggr = self.c_aggregate(x_list[2], x_node, edges_list[2][[1, 0], :], edge_weight_list[2])
        s_aggr = self.s_aggregate(x_list[3], x_node, edges_list[3][[1, 0], :], edge_weight_list[3])
        
        # Concatenate x_node with each aggregation
        g_aggr_cat = torch.cat((g_aggr, x_node), dim=-1)
        d_aggr_cat = torch.cat((d_aggr, x_node), dim=-1)
        c_aggr_cat = torch.cat((c_aggr, x_node), dim=-1)
        s_aggr_cat = torch.cat((s_aggr, x_node), dim=-1)
        
        # Matrix multiplication with learnable parameter u
        g_scores = torch.exp(F.leaky_relu(torch.matmul(g_aggr_cat, self.u)))
        d_scores = torch.exp(F.leaky_relu(torch.matmul(d_aggr_cat, self.u)))
        c_scores = torch.exp(F.leaky_relu(torch.matmul(c_aggr_cat, self.u)))
        s_scores = torch.exp(F.leaky_relu(torch.matmul(s_aggr_cat, self.u)))
        
        # Sum of scores for all types
        sum_scores = g_scores + d_scores + c_scores + s_scores
        
        # Calculate attention weights using softmax
        g_weights = g_scores / sum_scores
        d_weights = d_scores / sum_scores
        c_weights = c_scores / sum_scores
        s_weights = s_scores / sum_scores
        
        # Combine embeddings with attention weights
        combined_aggr = ((g_weights * g_aggr) +  (d_weights * d_aggr) + (c_weights * c_aggr) + (s_weights * s_aggr))
        
        # Concatenate the combined aggregation with x_node
        combined_aggr = torch.cat((x_node, combined_aggr), dim=-1)
        
        # Apply learnable linear layer to get the final aggregated embedding
        final_aggr = F.normalize(F.relu(self.linear(combined_aggr)), p=2, dim=-1)
        
        return final_aggr
 
 
class HAN_ConEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(HAN_ConEn, self).__init__()

        self.g_con = HAN_Content_Agg(embed_dim, 1, dropout)
        self.d_con = HAN_Content_Agg(embed_dim, 1, dropout)
        self.c_con = HAN_Content_Agg(embed_dim, 1, dropout)
        self.s_con = HAN_Content_Agg(embed_dim, 1, dropout)
        
        self.g_cont = torch.empty(0)
        self.d_cont = torch.empty(0)
        self.c_cont = torch.empty(0)
        self.s_cont = torch.empty(0)
        
    def forward(self, data):
        
        g_embed_list = [data['g'].x]
        d_embed_list = [data['d'].x]
        c_embed_list = [data['c'].x]
        s_embed_list = [data['s'].x]
        
        self.g_cont = self.g_con(g_embed_list)
        self.d_cont = self.d_con(d_embed_list)
        self.c_cont = self.c_con(c_embed_list)
        self.s_cont = self.s_con(s_embed_list)
        # print(g_embed_list[0].shape, d_embed_list[0].shape, c_embed_list[0].shape, s_embed_list[0].shape)
        
        return [self.g_cont, self.d_cont, self.c_cont, self.s_cont]
            

class HAN_NetEn(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super(HAN_NetEn, self).__init__()
        
        self.g_het = HAN_Agg(embed_dim, heads, dropout)
        self.d_het = HAN_Agg(embed_dim, heads, dropout)
        self.c_het = HAN_Agg(embed_dim, heads, dropout)
        self.s_het = HAN_Agg(embed_dim, heads, dropout)
        
    def forward(self, x_list, data):

        g_edges_list = [data['g', 'walk', 'g'].edge_index, data['g', 'walk', 'd'].edge_index, data['g', 'walk', 'c'].edge_index, data['g', 'walk', 's'].edge_index]
        d_edges_list = [data['d', 'walk', 'g'].edge_index, data['d', 'walk', 'd'].edge_index, data['d', 'walk', 'c'].edge_index, data['d', 'walk', 's'].edge_index]
        c_edges_list = [data['c', 'walk', 'g'].edge_index, data['c', 'walk', 'd'].edge_index, data['c', 'walk', 'c'].edge_index, data['c', 'walk', 's'].edge_index]
        s_edges_list = [data['s', 'walk', 'g'].edge_index, data['s', 'walk', 'd'].edge_index, data['s', 'walk', 'c'].edge_index, data['s', 'walk', 's'].edge_index]
        edge_weight_list = [None, None, None, None]
        
        x_list[1] = self.d_het(x_list, d_edges_list, x_list[1], edge_weight_list)
        x_list[0] = self.g_het(x_list, g_edges_list, x_list[0], edge_weight_list) 
        x_list[2] = self.c_het(x_list, c_edges_list, x_list[2], edge_weight_list)
        x_list[3] = self.s_het(x_list, s_edges_list, x_list[3], edge_weight_list)
        # print("NetEn:", x_list[0].shape, x_list[1].shape, x_list[2].shape, x_list[3].shape)
        
        return x_list
     

class HAN_classify(nn.Module):
    def __init__(self, embed_dim, heads, nclass, dropout):
        super(HAN_classify, self).__init__()
        self.d_het = HAN_Agg(embed_dim, heads, dropout)
        self.mlp = nn.Linear(embed_dim, nclass)    
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x_list, edge_list, edge_weight_list):
        x = self.d_het(x_list, edge_list, x_list[1], edge_weight_list)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        print("HAN_classify", x.shape)

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