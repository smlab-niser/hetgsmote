import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# Content Aggregation Layer
class HAN_Content_Agg(nn.Module):
    def __init__(self, embed_dim, attr_size, dropout):
        super(HAN_Content_Agg, self).__init__()
        self.fc = nn.Linear(attr_size, embed_dim)
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
        self.a_aggregate = HAN_NeighAgg(embed_dim, heads, dropout)
        self.p_aggregate = HAN_NeighAgg(embed_dim, heads, dropout)
        self.t_aggregate = HAN_NeighAgg(embed_dim, heads, dropout)
        self.c_aggregate = HAN_NeighAgg(embed_dim, heads, dropout)

        self.u = nn.Parameter(torch.randn(2 * embed_dim, 1))
        nn.init.normal_(self.u, mean=0, std=1)  # Normalize self.u
        self.linear = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, x_list, edges_list, x_node, edge_weight_list):
        a_aggr = self.a_aggregate(x_list[0], x_node, edges_list[0][[1, 0], :], edge_weight_list[0])
        p_aggr = self.p_aggregate(x_list[1], x_node, edges_list[1][[1, 0], :], edge_weight_list[1])
        t_aggr = self.t_aggregate(x_list[2], x_node, edges_list[2][[1, 0], :], edge_weight_list[2])
        c_aggr = self.c_aggregate(x_list[3], x_node, edges_list[3][[1, 0], :], edge_weight_list[3])
        
        # Concatenate x_node with each aggregation
        a_aggr_cat = torch.cat((a_aggr, x_node), dim=-1)
        p_aggr_cat = torch.cat((p_aggr, x_node), dim=-1)
        t_aggr_cat = torch.cat((t_aggr, x_node), dim=-1)
        c_aggr_cat = torch.cat((c_aggr, x_node), dim=-1)
        
        # Matrix multiplication with learnable parameter u
        a_scores = torch.exp(F.leaky_relu(torch.matmul(a_aggr_cat, self.u)))
        p_scores = torch.exp(F.leaky_relu(torch.matmul(p_aggr_cat, self.u)))
        t_scores = torch.exp(F.leaky_relu(torch.matmul(t_aggr_cat, self.u)))
        c_scores = torch.exp(F.leaky_relu(torch.matmul(c_aggr_cat, self.u)))
        
        # Sum of scores for all types
        sum_scores = a_scores + p_scores + t_scores + c_scores
        
        # Calculate attention weights using softmax
        a_weights = a_scores / sum_scores
        p_weights = p_scores / sum_scores
        t_weights = t_scores / sum_scores
        c_weights = c_scores / sum_scores
        
        # Combine embeddings with attention weights
        combined_aggr = ((a_weights * a_aggr) +  (p_weights * p_aggr) + (t_weights * t_aggr) + (c_weights * c_aggr))
        
        # Concatenate the combined aggregation with x_node
        combined_aggr = torch.cat((x_node, combined_aggr), dim=-1)
        
        # Apply learnable linear layer to get the final aggregated embedding
        final_aggr = F.normalize(F.relu(self.linear(combined_aggr)), p=2, dim=-1)
        
        return final_aggr
 
 
class HAN_ConEn(nn.Module):
    def __init__(self, embed_dim, args, dropout):
        super(HAN_ConEn, self).__init__()

        self.a_con = HAN_Content_Agg(embed_dim, args.A_emsize, dropout)
        self.p_con = HAN_Content_Agg(embed_dim, args.P_emsize, dropout)
        self.t_con = HAN_Content_Agg(embed_dim, args.T_emsize, dropout)
        self.c_con = HAN_Content_Agg(embed_dim, args.C_emsize, dropout)
        
        self.a_cont = torch.empty(0)
        self.p_cont = torch.empty(0)
        self.t_cont = torch.empty(0)
        self.c_cont = torch.empty(0)
        
    def forward(self, data):
        
        a_embed_list = [data['a'].x]
        p_embed_list = [data['p'].x]
        t_embed_list = [data['t'].x]
        c_embed_list = [data['c_embed'].x]
        
        self.a_cont = self.a_con(a_embed_list)
        self.p_cont = self.p_con(p_embed_list)
        self.t_cont = self.t_con(t_embed_list)
        self.c_cont = self.c_con(c_embed_list)
        
        return [self.a_cont, self.p_cont, self.t_cont, self.c_cont]
            

class HAN_NetEn(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super(HAN_NetEn, self).__init__()

        self.a_het = HAN_Agg(embed_dim, heads, dropout)
        self.p_het = HAN_Agg(embed_dim, heads, dropout)
        self.t_het = HAN_Agg(embed_dim, heads, dropout)
        self.c_het = HAN_Agg(embed_dim, heads, dropout)
        
    def forward(self, x_list, data):

        a_edges_list = [data['a', 'walk', 'a'].edge_index, data['a', 'walk', 'p'].edge_index, data['a', 'walk', 't'].edge_index, data['a', 'walk', 'c'].edge_index]
        p_edges_list = [data['p', 'walk', 'a'].edge_index, data['p', 'walk', 'p'].edge_index, data['p', 'walk', 't'].edge_index, data['p', 'walk', 'c'].edge_index]
        t_edges_list = [data['t', 'walk', 'a'].edge_index, data['t', 'walk', 'p'].edge_index, data['t', 'walk', 't'].edge_index, data['t', 'walk', 'c'].edge_index]
        c_edges_list = [data['c', 'walk', 'a'].edge_index, data['c', 'walk', 'p'].edge_index, data['c', 'walk', 't'].edge_index, data['c', 'walk', 'c'].edge_index]
        edge_weight_list = [None, None, None, None]
        
        x_list[0] = self.a_het(x_list, a_edges_list, x_list[0], edge_weight_list) 
        x_list[1] = self.p_het(x_list, p_edges_list, x_list[1], edge_weight_list)
        x_list[2] = self.t_het(x_list, t_edges_list, x_list[2], edge_weight_list)
        x_list[3] = self.c_het(x_list, c_edges_list, x_list[3], edge_weight_list)
        
        return x_list
    
    
class HAN_classify(nn.Module):
    def __init__(self, embed_dim, heads, nclass, dropout):
        super(HAN_classify, self).__init__()
        self.a_het = HAN_Agg(embed_dim, heads, dropout)
        self.mlp = nn.Linear(embed_dim, nclass)    
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x_list, edge_list, edge_weight_list):
        x = self.a_het(x_list, edge_list, x_list[0], edge_weight_list)
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