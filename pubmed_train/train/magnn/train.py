import torch
import torch.optim as optim
import torch.nn.functional as F
from smote import oversample
from metric import edge_loss, accuracy, evaluate_class_performance

w1 = 3
w2 = 3
w3 = 1
w4 = 1

# Train Function on the entire data
def train_smote(data, edge_indices, encoder1, encoder2, classifier, decoder_list, train_idx, 
                val_idx, test_idx, args, os_mode, train_mode):
    
    epochs = list(range(0, args.num_epochs, 10))

    val_acc_list, val_auc_list, val_f1_list = [], [], []
    test_acc_list, test_auc_list, test_f1_list = [], [], []
    loss_de, edge_ac = [], []
    auc_list, f1_list = [], []
    max_acc, max_auc, max_f1 = 0, 0, 0
    edge_weight_list = [None, None, None, None]

    # Define your optimizer for encoder and classifier
    optimizer_en1 = optim.Adam(encoder1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_en2 = optim.Adam(encoder2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_de = []

    encoder1.train()
    encoder2.train()
    classifier.train()
    
    for i in range(len(decoder_list)):
        optimizer_de.append(optim.Adam(decoder_list[i].parameters(), lr=args.lr, weight_decay=args.weight_decay))
        decoder_list[i].train()
    
    for epoch in range(args.num_epochs):

        optimizer_en1.zero_grad()
        optimizer_en2.zero_grad()
        optimizer_cls.zero_grad()
        for optimizer in optimizer_de:
            optimizer.zero_grad()
        
        x_list = encoder1(data)
        if os_mode == 'edge_sm' or os_mode == 'gsm' or os_mode == 'em_smote': x_list = encoder2(x_list, data)
            
        if os_mode == 'gsm':
            x_list[1], new_labels , new_train_idx, ar_edge_indices = oversample(features = x_list[1], labels = data['d'].y, 
                train_idx = train_idx, edge_indices = edge_indices, args= args, os_mode = 'gsm')
            
            edge_ac, loss_de, new_edge_indices, edge_weight_list = het_decode(decoder_list, x_list, edge_indices, ar_edge_indices, 
            args, train_mode, dataset = 'Train')
            loss_de = [loss_de[0]*w1, loss_de[1]*w2, loss_de[2]*w3, loss_de[3]*w4]
            del ar_edge_indices
                
        elif os_mode == 'embed_sm':
            x_list[1], new_labels , new_train_idx = oversample(features = x_list[1], labels = data['d'].y, 
                train_idx = train_idx, edge_indices = None, args= args, os_mode = 'gsm')
            new_edge_indices = edge_indices
            
        elif os_mode == 'up':
            x_list[1], new_labels , new_train_idx, new_edge_indices = oversample(features = x_list[1], labels =data['d'].y,  
            train_idx = train_idx, edge_indices = edge_indices, args= args, os_mode = 'up')
            
        elif os_mode == 'smote':
            x_list[1], new_labels , new_train_idx, new_edge_indices  = oversample(features = x_list[1], labels = data['d'].y, 
             train_idx = train_idx, edge_indices = edge_indices, args= args, os_mode = 'smote')
            
        elif os_mode == 'em_smote':
            x_list[1], new_labels , new_train_idx, new_edge_indices  = oversample(features = x_list[1], labels = data['d'].y, 
             train_idx = train_idx, edge_indices = edge_indices, args= args, os_mode = 'smote')
            
        else:
            new_labels, new_train_idx, new_edge_indices = data['d'].y, train_idx, edge_indices
            new_class_idx = [idx for idx, val in enumerate(new_labels[:,0]) if val.item() in new_train_idx]
        
        if os_mode != 'edge_sm' and os_mode != 'gsm' and os_mode != 'em_smote': x_list = encoder2(x_list, data)

        outputs = classifier(x_list, new_edge_indices, edge_weight_list, edge_index_mid1 = data['g', 'walk', 'c'].edge_index,  edge_index_mid2 = data['g', 'walk', 's'].edge_index)
        
        if os_mode != 'no' and os_mode != 'reweight': 
            new_class_idx = [idx for idx, val in enumerate(new_labels[:,0]) if val.item() in new_train_idx.tolist()]
        # print(len(new_class_idx), new_labels.shape, outputs.shape)
        
        if os_mode == 'reweight':
            weight = x_list[1].new((data['d'].y[:,1].max().item()+1)).fill_(1)
            for i, im in enumerate(args.im_class_num): weight[im] = 1+args.up_scale[i]
            loss_cls= F.cross_entropy(outputs[new_train_idx], new_labels[:,1][new_class_idx], weight=weight)
            del weight
        else:
            print("lossing")
            loss_cls = F.cross_entropy(outputs[new_train_idx], new_labels[:,1][new_class_idx])
        
        if train_mode == 'preO' or train_mode == 'preT' or train_mode == 'pret'  or train_mode == 'preo': 
            loss = sum(loss_de)*args.de_weight + loss_cls
            # print(loss_cls.item(), loss.item())
        elif train_mode == 'recon':
            loss = sum(loss_de) * args.de_weight * 10
        else:
            loss = loss_cls
        
        loss.backward()
        
        if train_mode != 'newG':
            optimizer_en1.step()
            optimizer_en2.step()
            if train_mode == 'preO' or train_mode == 'preT' or train_mode == 'recon' or train_mode == 'pret'  or train_mode == 'preo':
                for optimizer in optimizer_de:
                    optimizer.step()
                
        optimizer_cls.step()
        
        # if epoch in epochs:      
        acc = accuracy(outputs[new_train_idx].detach(), new_labels[:,1][new_class_idx].detach())
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, Edge Accuracy: {edge_ac}')
        evaluate_class_performance(outputs[new_train_idx].clone().detach(), new_labels[:,1][new_class_idx].clone().detach(), 
                                   thresh_list = [], dataset = 'Train', args = args)
        
        optimizer_en1.zero_grad()
        optimizer_en2.zero_grad()
        optimizer_cls.zero_grad()
        for optimizer in optimizer_de:
            optimizer.zero_grad()

        for x in x_list: del x
        del loss_cls
        del loss
        del new_labels
        del new_edge_indices
        del new_train_idx
        del outputs
            
        if val_idx is not None:
            ac, auc, f1, auc_scr, f1_scr = test_smote(data, edge_indices, encoder1, encoder2, classifier, decoder_list, 
            val_idx, args = args, dataset='Validation', os_mode='no', train_mode=train_mode)  
            
            if epoch in epochs:
                val_acc_list.append(ac)
                val_auc_list.append(auc)
                val_f1_list.append(f1)
        
        if test_idx is not None:
            ac, auc, f1, auc_scr, f1_scr = test_smote(data, edge_indices, encoder1, encoder2, classifier, decoder_list, 
            test_idx, args = args, dataset='Test', os_mode='no', train_mode=train_mode)  
            
            if max_acc < ac: max_acc = ac
            if max_auc < auc: max_auc = auc
            if max_f1 < f1: max_f1 = f1
            
            if epoch in epochs:
                test_acc_list.append(ac)
                test_auc_list.append(auc)
                test_f1_list.append(f1)
        
            if epoch == 149:
                auc_list = auc_scr
                f1_list = f1_scr
                 
            print()
    print("Finished Training")
    print("Validation Metrics:")
    print("Val_acc_list:", ["{:.4f}".format(val) for val in val_acc_list])
    print("Val_auc_list:", ["{:.4f}".format(val) for val in val_auc_list])
    print("Val_f1_list:", ["{:.4f}".format(val) for val in val_f1_list])
    print("Test Metrics:")
    print("Test_acc_list:", ["{:.4f}".format(val) for val in test_acc_list])
    print("Test_auc_list:", ["{:.4f}".format(val) for val in test_auc_list])
    print("Test_f1_list:", ["{:.4f}".format(val) for val in test_f1_list])
    
    test_acc_list.append(max_acc)
    test_auc_list.append(max_auc)
    test_f1_list.append(max_f1)
    
    return test_acc_list, test_auc_list, test_f1_list, auc_list, f1_list

# Test function
def test_smote(data, edge_indices, encoder1, encoder2, classifier, decoder_list, test_idx, os_mode, train_mode, args, dataset = "Test"):
    encoder1.eval()
    encoder2.eval()
    classifier.eval()
    for decoder in decoder_list: decoder.eval()
    loss_de, edge_ac = [], []
    auc_list, f1_list = [], []
    edge_weight_list = [None, None, None, None]
    
    if dataset == 'Validation': thresh_list = []
    else: thresh_list = args.best_threshold

    with torch.no_grad():
        x_list = encoder1(data)
        x_list = encoder2(x_list, data)
        
        if os_mode == 'gsm':
            edge_ac, loss_de, new_edge_indices, edge_weight_list = het_decode(decoder_list, x_list, edge_indices, 
                ar_edge_indices = None, args = args, train_mode = train_mode, dataset = dataset)
            loss_de = [loss_de[0]*w1, loss_de[1]*w2, loss_de[2]*w3, loss_de[3]*w4]
        else:
            new_edge_indices = edge_indices   
        
        outputs = classifier(x_list, new_edge_indices, edge_weight_list, edge_index_mid1 = data['g', 'walk', 'c'].edge_index,  edge_index_mid2 = data['g', 'walk', 's'].edge_index)
        new_class_idx = [idx for idx, val in enumerate(data['d'].y[:,0]) if val.item() in test_idx]
        
        loss_cls =  F.cross_entropy(outputs[test_idx], data['d'].y[:,1][new_class_idx])
        if train_mode == 'preO' or train_mode == 'preT' or train_mode == 'pret'  or train_mode == 'preo': 
            loss = sum(loss_de) * args.de_weight + loss_cls
        elif train_mode == 'recon':
            loss = torch.tensor(sum(loss_de))  
        else:
            loss = loss_cls            
        
        # print(loss)
        acc = accuracy(outputs[test_idx].detach(),data['d'].y[:,1][new_class_idx].detach()) 
        print(f'{dataset} Loss: {loss.item():.4f}, {dataset} Accuracy: {acc:.4f}, {dataset} Edge Accuracy: {edge_ac}')
        auc, f1, auc_list, f1_list = evaluate_class_performance(outputs[test_idx].clone().detach(), 
                    data['d'].y[:,1][new_class_idx].clone().detach(), thresh_list = thresh_list, dataset = dataset, args = args)
        
        for x in x_list: del x
        del loss_cls
        del loss
        del new_edge_indices
        
    return acc, auc, f1, auc_list, f1_list


# Function to evalute all decoder related tasks
def het_decode(decoder_list, x_list, edge_indices, ar_edge_indices, args, train_mode, dataset):
    edge_ac = []
    loss_de = []
    edge_list = []
    edge_weight_list = []
    acc = 0
 
    for i in range(len(decoder_list)):
        
        adj_new = decoder_list[i](x_list[1], x_list[i])
        ori_row_num = args.node_dim[1] # Changed from 0 to  in line 239, 240 and 247 since d is the second matrix in list
        ori_col_num = args.node_dim[i]
        
        adj_old = torch.zeros((ori_row_num, ori_col_num), dtype = torch.float32, device=x_list[0].device)
        adj_old[edge_indices[i][0], edge_indices[i][1]] = 1.0
        
        if ar_edge_indices is not None:
            adj_ar = torch.zeros((x_list[1].size(0), x_list[i].size(0)), dtype = torch.float32, device=x_list[0].device)
            adj_ar[ar_edge_indices[i][0], ar_edge_indices[i][1]] = 1.0
        else:
            adj_ar = None
  
        # edge_ac.append(F.l1_loss(adj_new[:ori_row_num, :ori_col_num], adj_old, reduction='mean'))
        loss_de.append(edge_loss(adj_new[:ori_row_num, :ori_col_num], adj_old))
        print(loss_de)

        adj_new, acc, edge_weight = adj_gen(adj_new, adj_old, adj_ar, ori_row_num, ori_col_num, dataset, train_mode)
        adj_new = adj_new.nonzero().t().contiguous()
        edge_ac.append(acc)
        
        if train_mode == 'preO' or train_mode == 'preo':
            edge_list.append(adj_new)
        else:
            edge_list.append(adj_new.detach())
        
        edge_weight_list.append(edge_weight)
            
    return edge_ac, loss_de, edge_list, edge_weight_list


def adj_gen(adj_new, adj_old, adj_ar, ori_row_num, ori_col_num, dataset, train_mode):
    threshold = 0.5
    edge_weight = None
    
    if adj_ar is not None:
        adj_new = torch.mul(adj_ar, adj_new)
        
    if train_mode == 'pret' or train_mode == 'preT': 
        adj_new = (adj_new >= threshold).float()

    diff = torch.abs(adj_new[:ori_row_num, :ori_col_num] - adj_old)
    correct_edges = (diff == 0).sum().item()
    total_edges = diff.numel()
    # print(correct_edges, total_edges)
    acc = correct_edges / total_edges
    # print(acc)
    
    if dataset == 'Train': adj_new[:ori_row_num, :ori_col_num] = adj_old
        
    if train_mode == 'preo' or train_mode == 'preO': 
        nonzero_indices = torch.nonzero(adj_new, as_tuple=False)
        edge_weight = adj_new[nonzero_indices[:, 0], nonzero_indices[:, 1]]  

    return adj_new, acc, edge_weight