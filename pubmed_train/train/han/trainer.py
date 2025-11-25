import sys
sys.path.extend([ '../', '../../','../../../'])
import torch
import dataloader as dl
from args import Args
from model import HAN_ConEn, HAN_NetEn, EdgePredictor, HAN_classify
from train import train_smote

# Set device to GPU if available, else use CPU
args = Args()
args.pubmed()
torch.cuda.empty_cache()

data = torch.load('../../data/pubmed_data.pt', weights_only=False)
file_path = '../output/im_ratio_han.txt'

device = args.device

# Send all x tensors to the device
data['g']['x'] = data['g']['x'].to(device)
data['d']['x'] = data['d']['x'].to(device)
data['c']['x'] = data['c']['x'].to(device)
data['s']['x'] = data['s']['x'].to(device)

# Send all y tensors to the device
data['d']['y'] = data['d']['y'].to(device)

data['g', 'walk', 'g']['edge_index'] = data['g', 'walk', 'g']['edge_index'].to(device) 
data['g', 'walk', 'd']['edge_index'] = data['g', 'walk', 'd']['edge_index'].to(device)
data['g', 'walk', 'c']['edge_index'] = data['g', 'walk', 'c']['edge_index'].to(device)
data['g', 'walk', 's']['edge_index'] = data['g', 'walk', 's']['edge_index'].to(device)

data['d', 'walk', 'g']['edge_index'] = data['d', 'walk', 'g']['edge_index'].to(device)
data['d', 'walk', 'd']['edge_index'] = data['d', 'walk', 'd']['edge_index'].to(device)
data['d', 'walk', 'c']['edge_index'] = data['d', 'walk', 'c']['edge_index'].to(device)
data['d', 'walk', 's']['edge_index'] = data['d', 'walk', 's']['edge_index'].to(device)

data['c', 'walk', 'g']['edge_index'] = data['c', 'walk', 'g']['edge_index'].to(device)
data['c', 'walk', 'd']['edge_index'] = data['c', 'walk', 'd']['edge_index'].to(device)
data['c', 'walk', 'c']['edge_index'] = data['c', 'walk', 'c']['edge_index'].to(device)
data['c', 'walk', 's']['edge_index'] = data['c', 'walk', 's']['edge_index'].to(device)

data['s', 'walk', 'g']['edge_index'] = data['s', 'walk', 'g']['edge_index'].to(device)
data['s', 'walk', 'd']['edge_index'] = data['s', 'walk', 'd']['edge_index'].to(device)
data['s', 'walk', 'c']['edge_index'] = data['s', 'walk', 'c']['edge_index'].to(device)
data['s', 'walk', 's']['edge_index'] = data['s', 'walk', 's']['edge_index'].to(device)

edge_indices = [ data['d', 'walk', 'g'].edge_index, data['d', 'walk', 'd'].edge_index, data['d', 'walk', 'c'].edge_index, data['d', 'walk', 's'].edge_index ]

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

up_ratios = [1.5, 1.6, 1.8, 2.0, 2.5]
im_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
r=0
z=13
# class_sample_nums = [50, 80, 100, 150]

with open(file_path, "w") as file:  
    
    for num, i in enumerate(im_ratios):
        
        args.im_ratio =  [0.7, 0.1, i, i, 0.4, 0.4]
        train_data_idx, val_data_idx, test_data_idx = [], [], []
        
        file.write(f'\nIm_ratio: {args.im_ratio}\n')
        
        if i == 0.5: 
            up_ratios = [1.5, 1.6, 1.8, 2.0, 2.5]
        else: 
            up_ratios = [0]
            z = 13
        
        for u, up in enumerate(up_ratios): 
            
            args.portion = up
            file.write(f'\nUp_ratio: {up}\n')
        
            for p in range(10):
                c_train_num = dl.train_num(data['d'].y[:,1], args.im_class_num, args.class_samp_num[0], args.im_ratio)
                print(c_train_num, sum(c_train_num))
                train_idx, val_idx, test_idx, c_num_mat = dl.segregate(data['d'].y, c_train_num, args.seed[1], args)
                
                train_data_idx.append(train_idx)
                val_data_idx.append(val_idx)
                test_data_idx.append(test_idx)
            
                
            for k in range(r, z):   
            
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
                    
                    classifier = HAN_classify(args.embed_dim, args.heads, args.nclass, args.dropout)

                    if k == 9 or k == 8:
                        encoder1 = torch.load('../pretrained_han/encoder1.pth', weights_only=False)
                        encoder2 = torch.load('../pretrained_han/encoder2.pth', weights_only=False)
                    else:
                        encoder1 = HAN_ConEn(args.embed_dim, args.dropout)
                        encoder2 = HAN_NetEn(args.embed_dim, args.heads, args.dropout)
                        
                    if train_dict[k] == 'preT' or train_dict[k] == 'preO' or train_dict[k] == 'noFT':
                        decoder_g = torch.load('../pretrained_han/decoder_g.pth', weights_only=False)
                        decoder_d = torch.load('../pretrained_han/decoder_d.pth', weights_only=False)
                        decoder_c = torch.load('../pretrained_han/decoder_c.pth', weights_only=False)
                        decoder_s = torch.load('../pretrained_han/decoder_s.pth', weights_only=False)
                    else: 
                        decoder_g = EdgePredictor(args.embed_dim)
                        decoder_d = EdgePredictor(args.embed_dim)
                        decoder_c = EdgePredictor(args.embed_dim)
                        decoder_s = EdgePredictor(args.embed_dim)
                
                    decoder_list = [decoder_g, decoder_d, decoder_c, decoder_s]
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
        
        
            