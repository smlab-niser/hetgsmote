import numpy as np
import random

def train_num(labels, im_class_num, class_sample_num, im_ratio):
    c_train_num = []
    max_class = labels.max().item()
    for i in range(max_class + 1):
        if i in im_class_num:                                              
            c_train_num.append(round(class_sample_num * im_ratio[im_class_num.index(i)]))
        else:
            c_train_num.append(class_sample_num)
    return c_train_num
    
    
    
# Function to randomly select the nodes for test, train and val based on the required samples from each class
def segregate(labels, c_train_num, seed, args):

    num_classes = max(labels.tolist())+1    
    c_idx = []                                         
    train_idx = []                                     
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)   
    c_num_mat[:,1] = args.class_samp_num[1]                                
    c_num_mat[:,2] = args.class_samp_num[2]
    random.seed(seed)
    
    for i in range(num_classes):                        
        
        c_idx = (labels==i).nonzero()[: ,-1].tolist()   
        print(i, len(c_idx))
        random.shuffle(c_idx)              
        
        train_idx = train_idx + c_idx[:c_train_num[i]]  
        c_num_mat[i,0] = c_train_num[i]                 

        val_idx = val_idx + c_idx[c_train_num[i]:c_train_num[i]+args.class_samp_num[1]]          
        test_idx = test_idx + c_idx[c_train_num[i]+args.class_samp_num[1]:c_train_num[i]+args.class_samp_num[1]+args.class_samp_num[2]]

    random.shuffle(train_idx)                           

    return train_idx, val_idx, test_idx, c_num_mat