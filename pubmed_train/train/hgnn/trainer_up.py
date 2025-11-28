import sys
sys.path.extend([ '../', '../../', '../../../'])
import torch
import dataloader as dl
from args import Args
from model import Het_ConEn, Het_NetEn, EdgePredictor, Het_classify
from train import train_smote

# Set device to GPU if available, else use CPU
args = Args()
args.dblp()
torch.cuda.empty_cache()

data = torch.load('../../data/dblp_data.pt')
file_path = '../output/up_scale_hgnn.txt'

device = args.device

# Send all x tensors to the device
data['a']['x'] = data['a']['x'].to(device)
data['p']['x'] = data['p']['x'].to(device)
data['t']['x'] = data['t']['x'].to(device)

data['c_embed']['x'] = data['c_embed']['x'].to(device)

# Send all y tensors to the device
data['a']['y'] = data['a']['y'].to(device)

data['a', 'walk', 'a']['edge_index'] = data['a', 'walk', 'a']['edge_index'].to(device) 
data['a', 'walk', 'p']['edge_index'] = data['a', 'walk', 'p']['edge_index'].to(device)
data['a', 'walk', 't']['edge_index'] = data['a', 'walk', 't']['edge_index'].to(device)
data['a', 'walk', 'c']['edge_index'] = data['a', 'walk', 'c']['edge_index'].to(device)

data['p', 'walk', 'a']['edge_index'] = data['p', 'walk', 'a']['edge_index'].to(device)
data['p', 'walk', 'p']['edge_index'] = data['p', 'walk', 'p']['edge_index'].to(device)
data['p', 'walk', 't']['edge_index'] = data['p', 'walk', 't']['edge_index'].to(device)
data['p', 'walk', 'c']['edge_index'] = data['p', 'walk', 'c']['edge_index'].to(device)

data['t', 'walk', 'a']['edge_index'] = data['t', 'walk', 'a']['edge_index'].to(device)
data['t', 'walk', 'p']['edge_index'] = data['t', 'walk', 'p']['edge_index'].to(device)
data['t', 'walk', 't']['edge_index'] = data['t', 'walk', 't']['edge_index'].to(device)
data['t', 'walk', 'c']['edge_index'] = data['t', 'walk', 'c']['edge_index'].to(device)

data['c', 'walk', 'a']['edge_index'] = data['c', 'walk', 'a']['edge_index'].to(device)
data['c', 'walk', 'p']['edge_index'] = data['c', 'walk', 'p']['edge_index'].to(device)
data['c', 'walk', 't']['edge_index'] = data['c', 'walk', 't']['edge_index'].to(device)
data['c', 'walk', 'c']['edge_index'] = data['c', 'walk', 'c']['edge_index'].to(device)

edge_indices = [ data['a', 'walk', 'a'].edge_index, data['a', 'walk', 'p'].edge_index, data['a', 'walk', 't'].edge_index, data['a', 'walk', 'c'].edge_index ]


train_dict = {
    0: 'no',
    1: 'up',
    2: 'smote',
    3: 'reweight',
    4: 'embed_sm',
    5: 'em_smote',
    6: 'pret',
    7: 'preo',
    8: 'preT', 
    9: 'preO',
    10: 'noFT',
    11: 'preT',
    12: 'preO'
} # 8, 9: pre enc + pre dec; 11,12: only pre dec

up_ratios = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

with open(file_path, "w") as file:   
    
    for num, up in enumerate(up_ratios):
        
        args.portion = up
        train_data_idx, val_data_idx, test_data_idx = [], [], []
        
        file.write(f'\nUp_ratio: {up}\n')
        
        for p in range(10):
            c_train_num = dl.train_num(data['a'].y, args.im_class_num, args.class_samp_num[0], args.im_ratio)
            print(c_train_num, sum(c_train_num))
            train_idx, val_idx, test_idx, c_num_mat = dl.segregate(data['a'].y, c_train_num, args.seed[p], args)
            
            train_data_idx.append(train_idx)
            val_data_idx.append(val_idx)
            test_data_idx.append(test_idx)
            
        for k in range(0, 13):   
        
            if k < 6:
                train_mode = ''
                os_mode = train_dict[k]
            else:
                train_mode = train_dict[k]
                os_mode = 'gsm'
            
            if k > 10: file.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}2\n')  
            else: file.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}\n')     
            file.flush()
            Test_acc, Test_auc, Test_f1, auc_list, f1_list = [], [], [], [], []
            
            for p in range(10):
                
                classifier = Het_classify(args.embed_dim, args.nclass, args.dropout)

                if k == 9 or k == 8:
                    encoder1 = torch.load('../pretrained_hgnn/encoder1.pth')
                    encoder2 = torch.load('../pretrained_hgnn/encoder2.pth')
                else:
                    encoder1 = Het_ConEn(args.embed_dim, args, args.dropout)
                    encoder2 = Het_NetEn(args.embed_dim, args.dropout)
                    
                if train_dict[k] == 'preT' or train_dict[k] == 'preO' or train_dict[k] == 'noFT':
                    decoder_a = torch.load('../pretrained_hgnn/decoder_a.pth')
                    decoder_p = torch.load('../pretrained_hgnn/decoder_p.pth')
                    decoder_t = torch.load('../pretrained_hgnn/decoder_t.pth')
                    decoder_c = torch.load('../pretrained_hgnn/decoder_c.pth')
                else: 
                    decoder_a = EdgePredictor(args.embed_dim)
                    decoder_p = EdgePredictor(args.embed_dim)
                    decoder_t = EdgePredictor(args.embed_dim)
                    decoder_c = EdgePredictor(args.embed_dim)
            
                decoder_list = [decoder_a, decoder_p, decoder_t, decoder_c]
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
        

            # file.write(f'\nClass Sample Num: {args.class_samp_num}\nTest_acc:\n')
            # file.write(f'\nUp_ratio: {args.portion}\nTest_acc:\n')
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
