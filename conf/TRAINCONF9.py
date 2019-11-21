"""data_modern_seg_5punc
"""
# data path
train_data = './data9_5punc_PFDSJ_seg/train'
cv_data = './data9_5punc_PFDSJ_seg/valid'
vocab = './data9_5punc_PFDSJ_seg/vocab'
punc_vocab = './data9_5punc_PFDSJ_seg/punc_vocab'
# train from prev checpoint
continue_from = ''
# 保存各个时间点的模型
save_folder = './exp/PRnet9'
# 是否在各个epoch结束保存检查点
checkpoint = True
# batch的尺寸:default=10
batch_size = 10
# L2:default=0.0
L2 = 0.0
# epoch大小:default=32
epochs = 32
# max_norm for clip gradient:default=250
max_norm = 250
# early_stop是否允许在epochs总数前结束训练
early_stop = True
# learning rate:default=1e-2
lr = 1e-2
# ########logging###########################################################
# verbose:true显示更多的信息
verbose = True
# print_freq:default=1000
print_freq = 10
# **************************************************************************
# ######### model hyper parameters##########################################
# 分类种类num_clss:default=3
num_class = 6
# lstm层数
num_layers = 3
# **************************************************************************
# ######## save and load model##############################################
model_path = 'final9.pth.tar'
