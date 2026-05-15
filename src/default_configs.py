import os
import json
import copy

params_ESD = {
   "naming":"ESD",
   "root_data_dir":"/home/zhangxiangning/",
   "dataset_name":"Multi_center",
   "sample_rate":1,
   "temporal_aug":True,
   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":11,
      "num_f_maps":256,
      "input_dim":2048,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.1,
      "temporal_dropout_rate":0.1,
      "feature_layer_indices":[9]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":128,
      "time_emb_dim":512,
      "kernel_size":5,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":8,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":[
         "zero",
         "full",
         "ant",
         "boundary03-",
         "segment=1",
         "segment=1"
      ],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.025,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.025,
      "decoder_boundary_loss":0.1
   },
   "batch_size":16,
   "learning_rate":0.0001,
   "weight_decay":0,
   "num_epochs":1001,
   "log_freq":20,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":3,
   "soft_label": 1,
   "log_train_results":False,
   "postprocess":{
      "type":"median",
      "value":25
   },
}


if __name__ == "__main__":
    params = copy.deepcopy(params_ESD)
    params['naming'] = 'ESD'

    if not os.path.exists('configs'):
        os.makedirs('configs')

    file_name = os.path.join('configs', f'{params["naming"]}.json')
    with open(file_name, 'w') as outfile:
        json.dump(params, outfile, ensure_ascii=False)
    print(f"Saved: {file_name}")
