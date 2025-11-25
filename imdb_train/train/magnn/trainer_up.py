import sys
sys.path.extend([ '../', '../../', '../../../'])
import torch
import dataloader as dl
from args import Args
from model import MAG_ConEn, MAG_NetEn, EdgePredictor,  MAG_classify
from train import train_smote

# Set device to GPU if available, else use CPU
args = Args()
args.imdb()
torch.cuda.empty_cache()

data = torch.load('../../data/data.pt', weights_only=False)
file_path = '../output/up_ratio_mag.txt'

device = args.device

# Send all x tensors to the device
data['m_text_embed']['x'] = data['m_text_embed']['x'].to(device)
data['m_net_embed']['x'] = data['m_net_embed']['x'].to(device)
data['m_a_net_embed']['x'] = data['m_a_net_embed']['x'].to(device)
data['m_d_net_embed']['x'] = data['m_d_net_embed']['x'].to(device)
data['a_net_embed']['x'] = data['a_net_embed']['x'].to(device)
data['a_text_embed']['x'] = data['a_text_embed']['x'].to(device)
data['d_net_embed']['x'] = data['d_net_embed']['x'].to(device)
data['d_text_embed']['x'] = data['d_text_embed']['x'].to(device)

# Send all y tensors to the device
data['m']['y'] = data['m']['y'].to(device)

# Send all edge_index tensors to the device
data['m', 'walk', 'm']['edge_index'] = data['m', 'walk', 'm']['edge_index'].to(device)
data['m', 'walk', 'a']['edge_index'] = data['m', 'walk', 'a']['edge_index'].to(device)
data['m', 'walk', 'd']['edge_index'] = data['m', 'walk', 'd']['edge_index'].to(device)
data['a', 'walk', 'm']['edge_index'] = data['a', 'walk', 'm']['edge_index'].to(device)
data['a', 'walk', 'a']['edge_index'] = data['a', 'walk', 'a']['edge_index'].to(device)
data['a', 'walk', 'd']['edge_index'] = data['a', 'walk', 'd']['edge_index'].to(device)
data['d', 'walk', 'm']['edge_index'] = data['d', 'walk', 'm']['edge_index'].to(device)
data['d', 'walk', 'a']['edge_index'] = data['d', 'walk', 'a']['edge_index'].to(device)
data['d', 'walk', 'd']['edge_index'] = data['d', 'walk', 'd']['edge_index'].to(device)

edge_indices = [ data['m', 'walk', 'm'].edge_index, data['m', 'walk', 'a'].edge_index, data['m', 'walk', 'd'].edge_index ]

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
} # 8, 9: pre enc + pre dec; 11,12: only pre dec

r, z = 0, 13

up_ratios = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
# up_ratios = [1.6, 1.8, 2.0, 2.2]

with open(file_path, "w") as file:   #, open(file_path2, "w") as file2:
    
    for num, up in enumerate(up_ratios):
        
        args.portion = up
        train_data_idx, val_data_idx, test_data_idx = [], [], []
        
        file.write(f'\nUp_ratio: {up}\n')
            
        for p in range(10):
            c_train_num = dl.train_num(data['m'].y, args.im_class_num, args.class_samp_num[0], args.im_ratio)
            print(c_train_num, sum(c_train_num))
            train_idx, val_idx, test_idx, c_num_mat = dl.segregate(data['m'].y, c_train_num, args.seed[p], args)
            
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
                    decoder_m = torch.load('../pretrained_magnn/decoder_m.pth', weights_only=False)
                    decoder_a = torch.load('../pretrained_magnn/decoder_a.pth', weights_only=False)
                    decoder_d = torch.load('../pretrained_magnn/decoder_d.pth', weights_only=False)
                else: 
                    decoder_m = EdgePredictor(args.embed_dim)
                    decoder_a = EdgePredictor(args.embed_dim)
                    decoder_d = EdgePredictor(args.embed_dim)
            
                decoder_list = [decoder_m, decoder_a, decoder_d]
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
    
        
            