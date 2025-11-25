import sys
sys.path.extend([ '../', '../../', '../../../'])
import torch
import dataloader as dl
from args import Args
from model import MAG_ConEn, MAG_NetEn, EdgePredictor, MAG_classify
from train import train_smote

# Set device to GPU if available, else use CPU
args = Args()
args.aminer_train()
torch.cuda.empty_cache()

data = torch.load('../../../data/aminer/am_data.pt', weights_only=False)
file_path = '../output/im_ratio_mag.txt'
args.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device = args.device
# device = torch.device("cpu")

# Send all x tensors to the device
data['p_title_embed']['x'] = data['p_title_embed']['x'].to(device)
data['p_abstract_embed']['x'] = data['p_abstract_embed']['x'].to(device)
data['p_net_embed']['x'] = data['p_net_embed']['x'].to(device)
data['p_a_net_embed']['x'] = data['p_a_net_embed']['x'].to(device)
data['p_p_net_embed']['x'] = data['p_p_net_embed']['x'].to(device)
data['p_v_net_embed']['x'] = data['p_v_net_embed']['x'].to(device)
data['a_net_embed']['x'] = data['a_net_embed']['x'].to(device)
data['a_text_embed']['x'] = data['a_text_embed']['x'].to(device)
data['v_net_embed']['x'] = data['v_net_embed']['x'].to(device)
data['v_text_embed']['x'] = data['v_text_embed']['x'].to(device)

# Send all y tensors to the device
data['a']['y'] = data['a']['y'].to(device)

# Send all edge_index tensors to the device
data['a', 'walk', 'a']['edge_index'] = data['a', 'walk', 'a']['edge_index'].to(device)
data['a', 'walk', 'p']['edge_index'] = data['a', 'walk', 'p']['edge_index'].to(device)
data['a', 'walk', 'v']['edge_index'] = data['a', 'walk', 'v']['edge_index'].to(device)
data['p', 'walk', 'a']['edge_index'] = data['p', 'walk', 'a']['edge_index'].to(device)
data['p', 'walk', 'p']['edge_index'] = data['p', 'walk', 'p']['edge_index'].to(device)
data['p', 'walk', 'v']['edge_index'] = data['p', 'walk', 'v']['edge_index'].to(device)
data['v', 'walk', 'a']['edge_index'] = data['v', 'walk', 'a']['edge_index'].to(device)
data['v', 'walk', 'p']['edge_index'] = data['v', 'walk', 'p']['edge_index'].to(device)
data['v', 'walk', 'v']['edge_index'] = data['v', 'walk', 'v']['edge_index'].to(device)

edge_indices = [ data['a', 'walk', 'a'].edge_index, data['a', 'walk', 'p'].edge_index, data['a', 'walk', 'v'].edge_index ]

train_dict = {
    0: 'no',
    1: 'up',
    2: 'smote',
    4: 'bsm',
    3: 'kmeans',
    5: 'reweight',
    6: 'embed_sm',
    7: 'em_smote',
    8: 'pret',
    9: 'preo',
    10: 'preT', 
    11: 'preO',
    12: 'noFT',
    13: 'preT',
    14: 'preO',
    15: 'gsm-b',
    16: 'gsm-k',
    17: 'gsm-b',
    18: 'gsm-k',
} 

r, z = 0, 13
up_ratios = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
class_sample_nums = [20, 30, 40, 50]

with open(file_path, "w") as file:   
    
    for n in range(len(class_sample_nums)): 
        
        samp = class_sample_nums[n]
        
        if n == 3: im_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else: im_ratios = [0.4]
        
        args.class_samp_num = [samp, samp+5, samp+int(samp/2)]
        
        file.write(f'\nClass Sample Num: {samp}\n')
        
        for num, i in enumerate(im_ratios):
            
            args.im_ratio = [i, 0.5, 0.6]
            file.write(f'\nIm_ratio: {args.im_ratio}\n')
            
            if num == 3: up_ratios = [0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
            else: up_ratios = [0]
            
            for u, up in enumerate(up_ratios):
                args.portion = up
                train_data_idx, val_data_idx, test_data_idx = [], [], []
                file.write(f'\nUp_ratio: {args.portion}\n')
                
                for p in range(10):
                    c_train_num = dl.train_num(data['a'].y, args.im_class_num, args.class_samp_num[0], args.im_ratio)
                    print(c_train_num, sum(c_train_num))
                    train_idx, val_idx, test_idx, c_num_mat = dl.segregate(data['a'].y, c_train_num, args.seed[p], args)
                    
                    train_data_idx.append(train_idx)
                    val_data_idx.append(val_idx)
                    test_data_idx.append(test_idx)
                    
                for k in range(r, z):   
                
                    if k < 8:
                        train_mode = ''
                        os_mode = train_dict[k]
                    elif k in [15, 16]:
                        train_mode = 'preT'
                        os_mode = train_dict[k]
                    elif k in [17, 18]:
                        train_mode = 'preO'
                        os_mode = train_dict[k]
                    else:
                        train_mode = train_dict[k]
                        os_mode = 'gsm'
                    
                    if k in [13,14]: file.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}2\n')  
                    else: file.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}\n')     
                    file.flush()
                    Test_acc, Test_auc, Test_f1, auc_list, f1_list = [], [], [], [], []
            
                    for p in range(10):
                            
                        classifier = MAG_classify(args.embed_dim, args.nclass, args.dropout)

                        if train_mode in ['preT', 'preO']:
                            encoder1 = torch.load('../pretrained_magnn/encoder1.pth', weights_only=False)
                            encoder2 = torch.load('../pretrained_magnn/encoder2.pth', weights_only=False)
                        else:
                            encoder1 = MAG_ConEn(args.embed_dim, args.dropout)
                            encoder2 = MAG_NetEn(args.embed_dim, args.dropout)
                            
                        if train_mode in ['preT', 'preO', 'noFT']:
                            decoder_a = torch.load('../pretrained_magnn/decoder_a.pth', weights_only=False)
                            decoder_p = torch.load('../pretrained_magnn/decoder_p.pth', weights_only=False)
                            decoder_v = torch.load('../pretrained_magnn/decoder_v.pth', weights_only=False)
                        else: 
                            decoder_a = EdgePredictor(args.embed_dim)
                            decoder_p = EdgePredictor(args.embed_dim)
                            decoder_v = EdgePredictor(args.embed_dim)
                    
                        decoder_list = [decoder_a, decoder_p, decoder_v]
                        #print(features.shape)
                        encoder1.to(device)
                        encoder2.to(device)
                        classifier.to(device)
                        for decoder in decoder_list:
                            decoder.to(device)
                            
                        train_idx, val_idx, test_idx = train_data_idx[p], val_data_idx[p], test_data_idx[p]
                    
                        test_acc_list, test_auc_list, test_f1_list, auc_cls_list, f1_cls_list = train_smote(data, edge_indices, encoder1, 
                        encoder2, classifier, decoder_list, train_idx, val_idx, test_idx, args, os_mode = os_mode, train_mode = train_mode)
                        Test_acc.append(test_acc_list)
                        Test_auc.append(test_auc_list)
                        Test_f1.append(test_f1_list)
                        auc_list.append(auc_cls_list)
                        f1_list.append(f1_cls_list)
                        torch.cuda.empty_cache()
                
                    file.write(f'\nTest_acc:\n')
                    for row in Test_acc:
                        row_str = " ".join(map(str, row))
                        file.write(row_str + "\n")
                    file.write(f'\nTest_auc:\n')    
                    for row in Test_auc:
                        row_str = " ".join(map(str, row))
                        file.write(row_str + "\n")
                    file.write(f'\nTest_f1:\n')  
                    for row in Test_f1:
                        row_str = " ".join(map(str, row))
                        file.write(row_str + "\n")
                    file.flush()
                    
                    file.write(f'\nClass AUC Score:\n')  
                    for row in auc_list:
                        row_str = " ".join(map(str, row))
                        file.write(row_str + "\n")
                    file.write(f'\nClass F1 Score:\n')  
                    for row in f1_list:  
                        row_str = " ".join(map(str, row))
                        file.write(row_str + "\n")
        
            