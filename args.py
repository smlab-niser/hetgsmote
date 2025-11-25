import torch

class Args:
    def __init__(self):
        # Default device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}" if torch.cuda.is_available() else "Current device: CPU")
        
        self.A_n, self.P_n, self.V_n = 0, 0, 0
        self.data_path = ''
        self.embed_dim = 0
        self.im_class_num = []
        self.im_ratio = []
        self.num_epochs = 0
        self.dropout = 0.1
        self.lr = 0.00001
        self.weight_decay = 5e-4
        self.portion = 1
        self.class_sample_num = 0
        self.nclass = 0
        self.node_dim = [self.A_n, self.P_n, self.V_n]
        self.de_weight = 0
        self.de_weight2 = 1e-6
        self.up_scale = 0
        self.thresh_dict = None
        self.best_threshold = []
        self.seed = list(range(10, 101, 10))
        self.heads = 8
        # self.device = torch.device("cpu")
        
    def aminer_train(self):
        self.A_n = 20171
        self.P_n = 13250
        self.V_n = 18
        self.data_path = ''
        self.embed_dim = 128
        self.im_class_num = [1,2,3]
        self.im_ratio = [0.6, 0.5, 0.4]
        self.num_epochs = 200
        self.portion = 0
        self.class_samp_num = [50, 55, 75] #50
        self.nclass = 4
        self.node_dim = [self.A_n, self.P_n, self.V_n]
        self.up_scale = [1/k for k in self.im_ratio]
        self.de_weight = 10
        self.lr = 0.0001
        self.thresh_dict = {
        0: [0.23, 0.21, 0.32, 0.30],
        1: [0.29, 0.23, 0.28, 0.31],
        2: [0.27, 0.26, 0.26, 0.31],
        3: [0.30, 0.25, 0.31, 0.31],
        4: [0.26, 0.28, 0.31, 0.28],
        5: [0.30, 0.26, 0.26, 0.33],
        6: [0.29, 0.23, 0.29, 0.33],
        7: [0.26, 0.23, 0.29, 0.29],
        8: [0.33, 0.25, 0.26, 0.31],
        9: [0.29, 0.26, 0.31, 0.26],
        10: [0.31, 0.28, 0.29, 0.26]}
        self.best_threshold = self.thresh_dict[0]
        
    
    def aminer_edge_pred(self):
        self.A_n = 20171
        self.P_n = 13250
        self.V_n = 18
        self.data_path = ''
        self.embed_dim = 128
        self.im_class_num = [1]
        self.im_ratio = [0.5]
        self.num_epochs = 200
        self.portion = 1/self.im_ratio[0]
        self.class_samp_num = [4000, 4000, 6000]
        self.nclass = 1
        self.node_dim = [self.A_n, self.P_n, self.V_n]
        self.up_scale = [1/k for k in self.im_ratio]
        self.de_weight = 10
        self.lr = 0.00001
        self.thresh_dict = {
        "no": [0.34],
        "up": [0.43],
        "smote": [0.43],
        "reweight": [0.49],
        "embed_sm": [0.40],
        "gsm": [0.41],
        "edge_sm": [0.43]}
        self.best_threshold = self.thresh_dict["no"]
        
    def aminer_whole(self):
        self.A_n = 28645
        self.P_n = 21044
        self.V_n = 18
        self.C_n = 4
        self.data_path = 'data/academic/'
        self.embed_dim = 128    
        
    def imdb(self):
        self.M_n = 4666
        self.A_n = 5845
        self.D_n = 2271
        self.embed_d = 128
        self.data_path = '../data/' 
        self.embed_dim = 128
        self.im_class_num = [0, 1]  # [0]
        self.im_ratio = [0.1, 0.1]#[0.4, 0.6]  # [0.5]
        self.num_epochs = 200
        self.portion = 0
        self.class_samp_num = [200, 140, 300] #[100, 140, 300]
        self.nclass = 3
        self.node_dim = [self.M_n, self.A_n, self.D_n]
        self.up_scale = [1/k for k in self.im_ratio]
        self.de_weight = 10
        self.lr = 0.0001   
        self.thresh_dict = {
        0: [0.26, 0.36, 0.30],
        1: [0.23, 0.44, 0.28],
        2: [0.28, 0.39, 0.29],
        3: [0.28, 0.38, 0.26],
        4: [0.29, 0.33, 0.29],
        5: [0.29, 0.34, 0.33],
        6: [0.26, 0.41, 0.29],
        7: [0.25, 0.43, 0.24],
        8: [0.28, 0.40, 0.30],
        9: [0.29, 0.46, 0.26],
        10: [0.25, 0.41, 0.28]}
        self.best_threshold = [0.33, 0.25, 0.26]
        
    def yelp(self):
        self.R_n = 157499
        self.U_n = 127228
        self.B_n = 5554
        self.embed_d = 128
        self.data_path = '../data/yelp_kaggle/' 
        self.embed_dim = 128
        self.im_class_num = [0]
        self.im_ratio =  [0.1] #[0.4, 0.6]
        self.num_epochs = 200
        self.portion = 0
        self.class_samp_num = [500, 600, 700]
        self.nclass = 3
        self.node_dim = [self.B_n, self.U_n, self.R_n]
        self.up_scale = 1/self.im_ratio[0]
        self.de_weight = 10
        self.lr = 0.0001   
        self.thresh_dict = {
        0: [0.26, 0.36, 0.30],
        1: [0.23, 0.44, 0.28],
        2: [0.28, 0.39, 0.29],
        3: [0.28, 0.38, 0.26],
        4: [0.29, 0.33, 0.29],
        5: [0.29, 0.34, 0.33],
        6: [0.26, 0.41, 0.29],
        7: [0.25, 0.43, 0.24],
        8: [0.28, 0.40, 0.30],
        9: [0.29, 0.46, 0.26],
        10: [0.25, 0.41, 0.28]}
        self.best_threshold = [0.33, 0.25, 0.26]
        
    def dblp(self):
        self.A_n = 4057
        self.P_n = 14328
        self.T_n = 7723
        self.C_n = 20
        self.A_emsize = 334
        self.P_emsize = 4231
        self.T_emsize = 50
        self.C_emsize = 128
        self.embed_d = 256
        self.data_path = '../data/' 
        self.embed_dim = 256
        self.im_class_num = [1, 2, 3]
        self.im_ratio =  [0.4, 0.5, 0.6] #[0.4, 0.6]
        self.num_epochs = 200
        self.portion = 0
        self.class_samp_num = [200, 50, 60]
        self.nclass = 4
        self.node_dim = [self.A_n, self.P_n, self.T_n, self.C_n]
        self.up_scale = [1/k for k in self.im_ratio]
        self.de_weight = 10
        self.lr = 0.0001   
        self.thresh_dict = {}
        self.best_threshold = [0.33, 0.25, 0.26, 0.30]
        
    def pubmed(self):
        self.G_n = 13561
        self.D_n = 20163
        self.C_n = 26522
        self.S_n = 2863
        self.embed_d = 200
        self.data_path = '../data/' 
        self.embed_dim = 200
        self.im_class_num = [2, 3, 4, 5, 6, 7]
        self.im_ratio =  [0.7, 0.1, 0.1, 0.1, 0.4, 0.4] #[0.4, 0.6]
        self.num_epochs = 200
        self.portion = 0
        self.class_samp_num = [16, 16, 20]
        self.nclass = 8
        self.node_dim = [self.G_n, self.D_n, self.C_n, self.S_n]
        self.up_scale = [1/k for k in self.im_ratio]
        self.de_weight = 10
        self.lr = 0.0001   
        self.thresh_dict = {}
        self.best_threshold = [0.33, 0.25, 0.26, 0.30]
        
        
    def movielens_edge_pred(self):
        self.U_n = 610
        self.M_n = 9742
        self.U_emsize = 512
        self.M_emsize = 660
        self.data_path = ''
        self.embed_dim = 256
        self.im_class_num = [0, 1, 2, 5]
        self.im_ratio = [0.1, 0.4, 0.5, 0.6]
        self.num_epochs = 2000
        self.portion = 0
        self.class_samp_num = [500, 600, 700]
        self.nclass = 6
        self.node_dim = [self.U_n, self.M_n]
        self.up_scale = [1/k for k in self.im_ratio]
        self.de_weight = 10
        self.lr = 0.00001
        self.thresh_dict = {}
        self.best_threshold = [0.33, 0.25, 0.26, 0.30, 0.31, 0.28]
    
    def mp_imdb(self):
        self.M_n = 4666
        self.embed_dim = 256
        self.data_path = 'data/' 
        self.im_class_num = [0, 1]  # [0]
        self.im_ratio = [0.4, 0.6]  # [0.5]
        self.num_epochs = 200
        self.portion = 0
        self.class_samp_num = [100, 140, 300]
        self.nclass = 3
        self.de_weight = 10
        self.lr = 0.0001   
        self.thresh_dict = {
        0: [0.26, 0.36, 0.30],
        1: [0.23, 0.44, 0.28],
        2: [0.28, 0.39, 0.29],
        3: [0.28, 0.38, 0.26],
        4: [0.29, 0.33, 0.29],
        5: [0.29, 0.34, 0.33],
        6: [0.26, 0.41, 0.29],
        7: [0.25, 0.43, 0.24],
        8: [0.28, 0.40, 0.30],
        9: [0.29, 0.46, 0.26],
        10: [0.25, 0.41, 0.28]}
        self.best_threshold = [0.33, 0.25, 0.26]
        
'''
no: [0.23, 0.21, 0.32, 0.30]
up: [0.29, 0.23, 0.28, 0.31]
smote: [0.27, 0.26, 0.26, 0.31]
reweight: [0.30, 0.25, 0.31, 0.31]
embed_sm: [0.26, 0.28, 0.31, 0.28]
no_rec_preT: [0.30, 0.26, 0.26, 0.33]
no_rec preO: [0.29, 0.23, 0.29, 0.33]
rec_preT: [0.26, 0.23, 0.29, 0.29]
rec_prO: [0.33, 0.25, 0.26, 0.31]
noFT: [0.29, 0.26, 0.31, 0.26]
newG: [0.31, 0.28, 0.29, 0.26]
'''
