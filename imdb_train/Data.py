import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from collections import Counter
        

class imdb:
    def __init__(self, args):                                  
        self.args = args
        self.content_filename = ["m_text_embed.txt", "node_net_embedding.txt"]
        
        self.m_text_embed = torch.zeros(args.M_n, args.embed_d, dtype=torch.float32)
        
        self.m_net_embed = torch.zeros(args.M_n, args.embed_d, dtype=torch.float32)
        self.a_net_embed = torch.zeros(args.A_n, args.embed_d, dtype=torch.float32)
        self.d_net_embed = torch.zeros(args.D_n, args.embed_d, dtype=torch.float32) 
        
        self.m_a_net_embed = torch.empty(0)
        self.m_d_net_embed = torch.empty(0)
        
        self.a_text_embed = torch.zeros(args.A_n, args.embed_d, dtype=torch.float32)
        self.d_text_embed = torch.zeros(args.D_n, args.embed_d, dtype=torch.float32)
        
        self.m_a_list = torch.empty(0)
        self.m_d_list = torch.empty(0)

        self.m_m_edge_index = torch.empty(0)
        self.m_a_edge_index = torch.empty(0)
        self.m_d_edge_index = torch.empty(0) 
        self.a_m_edge_index = torch.empty(0)
        self.a_a_edge_index = torch.empty(0)
        self.a_d_edge_index = torch.empty(0)
        self.d_m_edge_index = torch.empty(0)
        self.d_a_edge_index = torch.empty(0)
        self.d_d_edge_index = torch.empty(0)   
        
        self.m_class = torch.full((args.M_n,), -1, dtype=torch.long)
    
    def read_content_file(self): # m_text, m_net, a_net, d_net

        for f_name in self.content_filename:
            with open(self.args.data_path + f_name, 'r') as file:            
                lines = file.readlines()       
			
            for i, line in enumerate(lines):
                entries = line.strip().split()
                if f_name == 'm_text_embed.txt':
                    self.m_text_embed[i] = torch.tensor([float(x) for x in entries]) 
                else:
                    node_type = entries[0][0]
                    node_id = int(entries[0][1:])
                    if node_type == 'm':
                        self.m_net_embed[node_id] = torch.tensor([float(x) for x in entries[1:]])
                    elif node_type == 'a':
                        self.a_net_embed[node_id] = torch.tensor([float(x) for x in entries[1:]])
                    else:
                        self.d_net_embed[node_id] = torch.tensor([float(x) for x in entries[1:]])
        
    def read_cite_file(self): # m_a, m_d lists
 
        m_a_list = [] 
        m_d_list = []
        
        with open(self.args.data_path + "node_neighbours_all.txt", 'r') as file:            
            lines = file.readlines()[:self.args.M_n]

            for i, line in enumerate(lines):
                line = line.strip()
                node_id = int(re.split(':', line)[0][1:])
                neigh_list = re.split(',', re.split(':', line)[1].strip())
                
                for neigh in neigh_list:
                    neigh_id = neigh[0]
                    if neigh_id == "a":
                        m_a_list.append(torch.tensor([[node_id, int(neigh[1:])]]))
                    else:
                        m_d_list.append(torch.tensor([[node_id, int(neigh[1:])]]))

        # Concatenate the list of tensors into a single tensor
        self.m_a_list = torch.cat(m_a_list, dim=0).t().contiguous()
        self.m_d_list= torch.cat(m_d_list, dim=0).t().contiguous()
        
    def pt_aggr_embed(self): # m_a_net, m_d_net, a_text, d_text
        
        self.m_a_net_embed = aggregate(self.a_net_embed, self.m_a_list, self.args.M_n)
        self.m_d_net_embed = aggregate(self.d_net_embed, self.m_d_list, self.args.M_n)
        
        with open(self.args.data_path + "node_neighbours_all.txt", 'r') as file:            
            lines = file.readlines()[self.args.M_n:]
            
            for i, line in enumerate(lines):
                line = line.strip()
                node_type = re.split(':', line)[0][0]
                node_id = int(re.split(':', line)[0][1:])
                neigh_list = re.split(',', re.split(':', line)[1].strip())
                neigh_list = [int(neigh[1:]) for neigh in neigh_list]
                
                if len(neigh_list) >= 3:
                    if node_type == 'a':
                        self.a_text_embed[node_id] = torch.mean(self.m_text_embed[neigh_list[:3]], dim=0)
                    else:
                        self.d_text_embed[node_id] = torch.mean(self.m_text_embed[neigh_list[:3]], dim=0)
                else:
                    if node_type == 'a':
                        self.a_text_embed[node_id] = torch.mean(self.m_text_embed[neigh_list], dim=0)
                    else:
                        self.d_text_embed[node_id] = torch.mean(self.m_text_embed[neigh_list], dim=0)
    
    def read_walk_file(self): # all edges indices based on neighbours in random walks
        
        m_edge_list = []
        a_edge_list = []
        d_edge_list = []
        m_m_edge_index = []
        m_a_edge_index = []
        m_d_edge_index = []
        a_m_edge_index = []
        a_a_edge_index = []
        a_d_edge_index = []
        d_m_edge_index = []
        d_a_edge_index = []
        d_d_edge_index = []
        
        with open(self.args.data_path + "random_walks.txt", 'r') as file:            
            lines = file.readlines()
                
        for i, line in enumerate(lines):
            line = line.strip()
            node_type = re.split(':', line)[0][0]
            node_id = int(re.split(':', line)[0][1:])
            neigh_list = re.split(',', re.split(':', line)[1].strip())
            
            m_edge_list = [node for node in neigh_list if node.startswith('m')]
            a_edge_list = [node for node in neigh_list if node.startswith('a')]
            d_edge_list = [node for node in neigh_list if node.startswith('d')]
            
            # Count the frequency of elements starting with 'a', 'p', and 'v'
            m_counts = Counter(m_edge_list)
            a_counts = Counter(a_edge_list)
            d_counts = Counter(d_edge_list)
            # print(a_counts)
            
            m_edge_list = [node for node, count in m_counts.most_common(8)]
            a_edge_list = [node for node, count in a_counts.most_common(8)]
            d_edge_list = [node for node, count in d_counts.most_common(5)]
            # print(a_edge_list)
            
            if node_type == 'm':
                m_m_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in m_edge_list])
                m_a_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in a_edge_list])
                m_d_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in d_edge_list])
            elif node_type == 'a':
                a_m_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in m_edge_list])
                a_a_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in a_edge_list])
                a_d_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in d_edge_list])
            else:
                d_m_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in m_edge_list])
                d_a_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in a_edge_list])
                d_d_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in d_edge_list])
        
        # Concatenate the list of tensors into a single tensor
        self.m_m_edge_index = torch.cat(m_m_edge_index, dim=0).t().contiguous()
        self.m_a_edge_index = torch.cat(m_a_edge_index, dim=0).t().contiguous()
        self.m_d_edge_index = torch.cat(m_d_edge_index, dim=0).t().contiguous()
        self.a_m_edge_index = torch.cat(a_m_edge_index, dim=0).t().contiguous()
        self.a_a_edge_index = torch.cat(a_a_edge_index, dim=0).t().contiguous()
        self.a_d_edge_index = torch.cat(a_d_edge_index, dim=0).t().contiguous()
        self.d_m_edge_index = torch.cat(d_m_edge_index, dim=0).t().contiguous()
        self.d_a_edge_index = torch.cat(d_a_edge_index, dim=0).t().contiguous()
        self.d_d_edge_index = torch.cat(d_d_edge_index, dim=0).t().contiguous()
        
        
    def read_label_file(self):
        
        with open(self.args.data_path + "m_class.txt", 'r') as file:            
            lines = file.readlines()
        for i, line in enumerate(lines):
                entries =  line.strip().split(',')
                self.m_class[int(entries[0])] = int(entries[1].strip())
 
       
def input_data(args):
    dataset = imdb(args)
    dataset.read_content_file()
    dataset.read_cite_file()
    dataset.pt_aggr_embed()
    dataset.read_walk_file()
    dataset.read_label_file()

    data = HeteroData()
    
    data['m'].num_nodes = args.M_n
    data['a'].num_nodes = args.A_n
    data['d'].num_nodes = args.D_n

    data['m_text_embed'].x = dataset.m_text_embed
    data['m_net_embed'].x = dataset.m_net_embed
    data['m_a_net_embed'].x = dataset.m_a_net_embed
    data['m_d_net_embed'].x = dataset.m_d_net_embed
        
    data['a_net_embed'].x = dataset.a_net_embed
    data['a_text_embed'].x = dataset.a_text_embed
        
    data['d_net_embed'].x = dataset.d_net_embed
    data['d_text_embed'].x = dataset.d_text_embed
    
    data['m'].y = dataset.m_class

    data['m', 'walk', 'm'].edge_index = dataset.m_m_edge_index
    data['m', 'walk', 'a'].edge_index = dataset.m_a_edge_index
    data['m', 'walk', 'd'].edge_index = dataset.m_d_edge_index
    
    data['a', 'walk', 'm'].edge_index = dataset.a_m_edge_index
    data['a', 'walk', 'a'].edge_index = dataset.a_a_edge_index
    data['a', 'walk', 'd'].edge_index = dataset.a_d_edge_index
    
    data['d', 'walk', 'm'].edge_index = dataset.d_m_edge_index
    data['d', 'walk', 'a'].edge_index = dataset.d_a_edge_index
    data['d', 'walk', 'd'].edge_index = dataset.d_d_edge_index
    
    return data
    
 
    



# Function for heterogenous neighbour aggregation
def aggregate(x, edge_index, num_nodes): 
    # Separate source and target nodes from the edge index
    source_nodes, target_nodes = edge_index[0], edge_index[1]

    # Aggregate features for each neighbour using scatter_add
    # num_source = torch.max(source_nodes, dim = 0).values.item()
    aggr_features = torch.zeros(num_nodes, x.size(1))
    aggr_features.index_add_(0, source_nodes, x[target_nodes])

    # Normalize the aggregated features
    row_sum = torch.bincount(source_nodes, minlength=num_nodes).float().clamp(min=1)
    aggr_features /= row_sum.view(-1, 1)

    return aggr_features