import torch

train_config = {}
# train_config['traj_base_path'] = '/home/lgeng/cogen/data'
train_config['data_name'] = 'pour_60'
train_config['set_type'] = 'train'
train_config['success_data_path'] = '/home/lgeng/hrdc/pour_extracted/Pick_and_Place/cool/Env1'
train_config['unsuccess_data_path'] = '/home/lgeng/hrdc/pour_failure_extracted/Door_Opening/my_home/Env1'

valid_config = {}
# valid_config['traj_base_path'] = '/home/lgeng/cogen/data'
valid_config['data_name'] = 'new_pour_60'
valid_config['set_type'] = 'valid'
valid_config['success_data_path'] = '/home/lgeng/hrdc/pour_extracted/Pick_and_Place/cool/Env1'
valid_config['unsuccess_data_path'] = '/home/lgeng/hrdc/new_pour_failure_extracted/Door_Opening/my_home/Env1'

debug = True
image_path = "/home/lgeng/contrCoformer/Images"
captions_path = "/home/lgeng/contrCoformer"
batch_size = 4
num_workers = 0
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet50'
image_embedding = 2048
text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 400

pretrained = False # for both image encoder and text encoder
trainable = False # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1