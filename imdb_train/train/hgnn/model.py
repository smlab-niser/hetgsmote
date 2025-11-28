import torch
import torch.nn as nn
import torch.nn.functional as F

class Content_Agg(nn.Module):
    def __init__(self, embed_dim, num_embed, dropout):
        super(Content_Agg, self).__init__()
        self.fc = nn.Linear(embed_dim * num_embed, embed_dim)
        self.dropout = dropout

    def forward(self, embed_list):
        # Concatenate the embeddings
        x = torch.cat(embed_list, dim=-1)
        x = F.normalize(F.relu(self.fc(x)), p=2, dim = -1)
        x = F.dropout(x, self.dropout, training=self.training) 
        return x


# Layer for heterogenous neighbour aggregation
class Neigh_Agg(nn.Module): 
    def __init__(self, embed_dim, dropout):
        super(Neigh_Agg, self).__init__()
        self.aggregation_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index, num_node, edge_weight = None):
        
        if edge_weight is None: edge_weight = torch.ones(edge_index.size(1), device=x.device)
        
        # Separate source and target nodes from the edge index
        source_nodes, target_nodes = edge_index[0], edge_index[1]

        # Apply ReLU activation to target features through the linear layer
        x_target = F.relu(self.aggregation_layer(x))
        
        # Multiply each x_target element by the respective edge_weights element
        weighted_x_target = x_target[target_nodes] * edge_weight.view(-1, 1)

        # Aggregate target features for each source using scattea_add
        aggr_features = torch.zeros(num_node, x.size(1), device=x.device)
        aggr_features.index_add_(0, source_nodes, weighted_x_target)

        # Normalize the aggregated features
        row_sum = torch.bincount(source_nodes, minlength=num_node).float().clamp(min=1)
        aggr_features /= row_sum.view(-1, 1)
        aggr_features = F.dropout( aggr_features, self.dropout, training=self.training)

        return aggr_features
    

class Het_Agg(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Het_Agg, self).__init__()
        self.m_aggregate = Neigh_Agg(embed_dim, dropout)
        self.a_aggregate = Neigh_Agg(embed_dim, dropout)
        self.d_aggregate = Neigh_Agg(embed_dim, dropout)

        self.u = nn.Parameter(torch.randn(2 * embed_dim, 1))
        nn.init.normal_(self.u, mean=0, std=1)  # Normalize self.u
        self.linear = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, x_list, edges_list, x_node, num_node, edge_weight_list):
        m_aggr = self.m_aggregate(x_list[0], edges_list[0], num_node, edge_weight_list[0])
        a_aggr = self.a_aggregate(x_list[1], edges_list[1], num_node, edge_weight_list[1])
        d_aggr = self.d_aggregate(x_list[2], edges_list[2], num_node, edge_weight_list[2])
        
        # Concatenate x_node with each aggregation
        m_aggr_cat = torch.cat((m_aggr, x_node), dim=-1)
        a_aggr_cat = torch.cat((a_aggr, x_node), dim=-1)
        d_aggr_cat = torch.cat((d_aggr, x_node), dim=-1)
        
        # Matrix multiplication with learnable parameter u
        m_scores = torch.exp(F.leaky_relu(torch.matmul(m_aggr_cat, self.u)))
        a_scores = torch.exp(F.leaky_relu(torch.matmul(a_aggr_cat, self.u)))
        d_scores = torch.exp(F.leaky_relu(torch.matmul(d_aggr_cat, self.u)))
        
        # Sum of scores for all types
        sum_scores = m_scores + a_scores + d_scores
        
        # Calculate attention weights using softmax
        m_weights = m_scores / sum_scores
        a_weights = a_scores / sum_scores
        d_weights = d_scores / sum_scores
        
        # Combine embeddings with attention weights
        combined_aggr = ((m_weights * m_aggr) +  (a_weights * a_aggr) + (d_weights * d_aggr))
        
        # Concatenate the combined aggregation with x_node
        combined_aggr = torch.cat((x_node, combined_aggr), dim=-1)
        
        # Apply learnable linear layer to get the final aggregated embedding
        final_aggr = F.normalize(F.relu(self.linear(combined_aggr)), p=2, dim=-1)
        
        return final_aggr
 
 
class Het_ConEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Het_ConEn, self).__init__()

        self.m_con = Content_Agg(embed_dim, 4, dropout)
        self.a_con = Content_Agg(embed_dim, 2, dropout)
        self.d_con = Content_Agg(embed_dim, 2, dropout)
        
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
            

class Het_NetEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Het_NetEn, self).__init__()
        
        self.m_het = Het_Agg(embed_dim, dropout)
        self.a_het = Het_Agg(embed_dim, dropout)
        self.d_het = Het_Agg(embed_dim, dropout)
        
    def forward(self, x_list, data):

        m_edges_list = [data['m', 'walk', 'm'].edge_index, data['m', 'walk', 'a'].edge_index, data['m', 'walk', 'd'].edge_index]
        a_edges_list = [data['a', 'walk', 'm'].edge_index, data['a', 'walk', 'a'].edge_index, data['a', 'walk', 'd'].edge_index]
        d_edges_list = [data['d', 'walk', 'm'].edge_index, data['d', 'walk', 'a'].edge_index, data['d', 'walk', 'd'].edge_index]
        edge_weight_list = [None, None, None]
        
        x_list[0] = self.m_het(x_list, m_edges_list, x_list[0], x_list[0].size(0), edge_weight_list) 
        x_list[1] = self.a_het(x_list, a_edges_list, x_list[1], x_list[1].size(0), edge_weight_list)
        x_list[2] = self.d_het(x_list, d_edges_list, x_list[2], x_list[2].size(0), edge_weight_list)
        
        return x_list


class Het_classify(nn.Module):
    def __init__(self, embed_dim, nclass, dropout):
        super(Het_classify, self).__init__()
        self.m_het = Het_Agg(embed_dim, dropout)
        self.mlp = nn.Linear(embed_dim, nclass)    
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x_list, edge_list, edge_weight_list):
        x = self.m_het(x_list, edge_list, x_list[0], x_list[0].size(0), edge_weight_list)
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
    
    