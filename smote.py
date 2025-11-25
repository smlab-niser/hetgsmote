import torch
import numpy as np
import random
from scipy.spatial.distance import pdist,squareform
from imblearn.over_sampling import BorderlineSMOTE, KMeansSMOTE, SMOTE
from sklearn.metrics import pairwise_distances

'''
train_idx: Indices of the nodes used for training
portion: Number of times the nodes of a class is to be sampled
im_class_num: The upper cap class number of the imbalanced classes
os_mode: oversampling type
'''
def oversample(features, labels, train_idx, edge_indices, args, os_mode):  
    
    train_idx = torch.LongTensor(train_idx).to(args.device)
    max_class = labels.max().item()         # max_class stores the largest class number
    new_features = None                     # Stores feature matrix of the synthetic nodes
    sample_idx_list = []
    near_neighbor_list = []

    # avg_number = int(train_idx.shape[0]/(max_class+1))   # Stores the average number of nodes per class
    avg_number = args.class_samp_num[0]

    for i in args.im_class_num:    # Loop interates over all imbalanced classes to apply smote on

        sample_idx = train_idx[(labels==i)[train_idx]]    # Stores the indices from train_idx that belong to class max_class - i
        total_count = 0
        ini_count = len(sample_idx)

        if args.portion == 0: #refers to even distribution
            c_portion = int(avg_number/sample_idx.shape[0])   # c_portion holds the integer times the sample_idx are to be sampled to reach average number of nodes
            portion_rest = (avg_number/sample_idx.shape[0]) - c_portion # Stores the leftover fraction of the sampling to be done

        else:
            c_portion = int(args.portion)
            portion_rest = args.portion - c_portion
        
        for j in range(c_portion):  # This loops runs about the number of times of upsampling of all nodes + fractional
            
            if j == c_portion - 1: # When only the leftover fractional part of total sampling is left
                if portion_rest == 0.0: break
                left_portion = round(sample_idx.shape[0]*portion_rest)  # stores the number of indices to be sampled
                sample_idx = sample_idx[:left_portion]          # Stores the subset of indices to be sampled
                if left_portion == 0: break

            samples = features[sample_idx, :]    # Picks out the sample_idx nodes feature vector from features
            if not j: total_count = len(sample_idx)
            
            if os_mode == 'up':
                new_samples = samples

            elif os_mode == 'bsm':
                X = samples.cpu().detach().numpy()  # Convert to numpy
                y = labels[sample_idx].cpu().detach().numpy()
                total_count += X.shape[0]
                
                full_X = features[train_idx].cpu().detach().numpy() 
                full_y = labels[train_idx].cpu().detach().numpy()
        
                sm = BorderlineSMOTE(sampling_strategy={i: total_count}, k_neighbors=5)
                X_res, y_res = sm.fit_resample(full_X, full_y)

                # Extract only the new samples
                new_X = X_res[len(full_X)+total_count-ini_count-len(samples):]
                new_samples = torch.tensor(new_X, dtype=features.dtype).to(args.device)

                # Find nearest neighbors *from* original class samples
                ref_X = samples.cpu().detach().numpy()
                dists = pairwise_distances(new_X, ref_X)
                neighbors = dists.argmin(axis=1)
                near_neighbor = [sample_idx[n].item() for n in neighbors]

            elif os_mode == 'kmeans':
                X = samples.cpu().detach().numpy()  # Convert to numpy
                y = labels[sample_idx].cpu().detach().numpy()
                total_count += X.shape[0]
                
                full_X = features[train_idx].cpu().detach().numpy() 
                full_y = labels[train_idx].cpu().detach().numpy()
                # print(ini_count)

                if ini_count < 50:
                    cb_thresh = 0.05  # no cluster balance enforced
                else:
                    cb_thresh = 0.1

                sm = KMeansSMOTE(sampling_strategy={i: total_count}, cluster_balance_threshold=cb_thresh)
                X_res, y_res = sm.fit_resample(full_X, full_y)

                # Extract only the new samples
                new_X = X_res[len(full_X)+total_count-ini_count-len(samples):][:X.shape[0]]
                new_samples = torch.tensor(new_X, dtype=features.dtype).to(args.device)

                # Find nearest neighbors *from* original class samples
                ref_X = samples.cpu().detach().numpy()
                dists = pairwise_distances(new_X, ref_X)
                neighbors = dists.argmin(axis=1)
                near_neighbor = [sample_idx[n].item() for n in neighbors]

            else:
                distance = squareform(pdist(samples.cpu().detach()))  # Calculates distance between each pair of nodes into square symmetric matrix A[i,j] = Distance between i-th and j-th node
                np.fill_diagonal(distance,distance.max() + 100)       # Make the diagonal element which would be 0 as maximum since we can't interpolate between the same node
                near_neighbor = distance.argmin(axis = -1)            # Stores the indices of the nearest neighbour in each row for each node. These pairs of nodes will be interpolated on.
                interp_place = random.random()                        # Define weight for interpolation
                new_samples = samples + (samples[near_neighbor,:] - samples)*interp_place # Interpolation: X = A + (B-A)*W
            
            if new_features is None: new_features = new_samples
            else:  new_features = torch.cat((new_features, new_samples),0)  
            
            sample_idx_list.extend(sample_idx)  
            if os_mode != 'up':          
                near_neighbor_list.extend(near_neighbor)
        
        if new_features is not None:   
            new_labels = labels.new(torch.Size((new_features.shape[0],1))).reshape(-1).fill_(i) # Make a new tensor of the same type as labels tensor and fill it with class max_class - i
            new_idx = np.arange(features.shape[0], features.shape[0] + new_features.shape[0]) # Stores new set of indices 
            new_train_idx = train_idx.new(new_idx)                                            
        
            features = torch.cat((features , new_features), 0)      # Update feature matrix, labels, and train_idx
            labels = torch.cat((labels, new_labels), 0)
            train_idx = torch.cat((train_idx, new_train_idx), 0)
            # print(features.shape, labels.shape, train_idx.shape)

            new_features = None

    if edge_indices is not None:
        if os_mode in ['smote', 'gsm', 'bsm', 'kmeans']:
            edge_list = sm_edge_gen(sample_idx_list, near_neighbor_list, edge_indices, args)
        else:
            edge_list = up_edge_gen(sample_idx_list, edge_indices, args)
        return features, labels, train_idx, edge_list
    else:
        return features, labels, train_idx

def up_edge_gen(sample_idx_list, edge_indices, args):     
    edge_list = []       
    add_num = len(sample_idx_list)    
    
    for i in range(len(edge_indices)):
        adj = torch.zeros((args.node_dim[0], args.node_dim[i]), dtype = torch.float32).to(args.device)
        adj[edge_indices[i][0], edge_indices[i][1]] = 1.0
        if i == 0:
            new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1]+add_num)))
            new_adj[:adj.shape[0], adj.shape[1]:] = adj[:,sample_idx_list]
            new_adj[adj.shape[0]:, adj.shape[1]:] = adj[sample_idx_list,:][:,sample_idx_list]
        else:
            new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1])))
        new_adj[:adj.shape[0], :adj.shape[1]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[1]] = adj[sample_idx_list,:]
        edge_list.append(new_adj.nonzero().t().contiguous().to(args.device))
        
    return edge_list


def sm_edge_gen(sample_idx_list, near_neighbors, edge_indices, args):     
    edge_list = []       
    add_num = len(sample_idx_list)   
    
    for i in range(len(edge_indices)):
        adj = torch.zeros((args.node_dim[0], args.node_dim[i]), dtype = torch.float32).to(args.device)
        adj[edge_indices[i][0], edge_indices[i][1]] = 1.0
        
        if i == 0:
            new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1]+add_num)))
            adj_new = adj.new(torch.clamp_(adj[:,sample_idx_list] + adj[:,near_neighbors], min=0.0, max = 1.0))
            new_adj[:adj.shape[0], adj.shape[1]:] = adj_new[:,:]
            adj_new = adj.new(torch.clamp_(adj[sample_idx_list,:][:,sample_idx_list] + adj[near_neighbors,:][:,near_neighbors], min=0.0, max = 1.0))
            new_adj[adj.shape[0]:, adj.shape[1]:] = adj_new[:,:]
            
        else:
            new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1])))
        
        adj_new = adj.new(torch.clamp_(adj[sample_idx_list,:] + adj[near_neighbors,:], min=0.0, max = 1.0))
        new_adj[:adj.shape[0], :adj.shape[1]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[1]] = adj_new[:,:]
        edge_list.append(new_adj.nonzero().t().contiguous().to(args.device))
        
    return edge_list