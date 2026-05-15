import os
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import median_filter
import numpy as np
from tqdm import tqdm
from .dataset import restore_full_sequence
from .model.actfusion import ActFusion
from .utils import func_eval, get_labels_start_end_time, mode_filter, read_mapping_dict, eval_file, get_unique_list, get_unique_list_gt, eval
from .vis import segment_bars

class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params,
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess

        self.model = ActFusion(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        
        # Initialize best performance tracking
        self.best_tas_acc = 0.0
        self.best_lta_moc = 0.0
        self.best_both_score = 0.0
        self.best_tas_metrics = None
        self.best_lta_metrics = None
        self.best_combined_metrics = None

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, multi_center_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq, log_train_results=True, all_params=None):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        if os.path.exists(result_dir):
            if 'latest.pth' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pth')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pth'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')

        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)

        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)

        for epoch in range(restore_epoch+1, num_epochs):
            self.model.train()
            epoch_running_loss = 0
            for _, data in enumerate(train_train_loader):
                feature, label, boundary, video = data
                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)

                loss_dict = self.model.get_training_loss(feature,
                    event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    boundary_gt=boundary,
                    encoder_ce_criterion=ce_criterion,
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )


                # ##############
                # # feature    torch.Size([1, F, T])
                # # label      torch.Size([1, T])
                # # boundary   torch.Size([1, 1, T])
                # # output    torch.Size([1, C, T])
                # ##################

                total_loss = 0

                for k, v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k, v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()

                epoch_running_loss += total_loss.item()

                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1

            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')

            if result_dir:

                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:

                w_path = os.path.join(result_dir, 'log.txt')
                w = open(w_path, 'a')
                w_ = 'epoch'+str(epoch)+'\n'

                if result_dir:
                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.pth')
                    torch.save(state, f'{result_dir}/latest.pth')

                # for mode in ['encoder', 'decoder-noagg', 'decoder-agg']:
                for mode in ['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar
                    # Test Dataset Segmentation (TAS) inference
                    # TAS inference
                    test_result_dict = self.test(
                        test_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=1.0)
                    w_ = self._log_tas_results(test_result_dict, w_)

                    # LTA inference - obs_p 0.2
                    test_result_dict20 = self.test(
                        test_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=0.2)
                    w_ = self._log_lta_results(test_result_dict20, w_, obs_p=0.2)

                    # LTA inference - obs_p 0.3
                    test_result_dict30 = self.test(
                        test_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=0.3)
                    w_ = self._log_lta_results(test_result_dict30, w_, obs_p=0.3)

                    # LTA inference - obs_p 0.5
                    test_result_dict50 = self.test(
                        test_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=0.5)
                    w_ = self._log_lta_results(test_result_dict50, w_, obs_p=0.5)

                    # Multi-center Dataset Segmentation (TAS) inference
                    # TAS inference
                    multi_center_result_dict = self.test(
                        multi_center_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=1.0)
                    w_ = self._log_tas_results(multi_center_result_dict, w_)

                    # LTA inference - obs_p 0.2
                    multi_center_result_dict20 = self.test(
                        multi_center_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=0.2)
                    w_ = self._log_lta_results(multi_center_result_dict20, w_, obs_p=0.2)

                    # LTA inference - obs_p 0.3
                    multi_center_result_dict30 = self.test(
                        multi_center_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=0.3)
                    w_ = self._log_lta_results(multi_center_result_dict30, w_, obs_p=0.3)

                    # LTA inference - obs_p 0.5
                    multi_center_result_dict50 = self.test(
                        multi_center_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=0.5)
                    w_ = self._log_lta_results(multi_center_result_dict50, w_, obs_p=0.5)


                    # === Best model saving ===
                    tas_score = (test_result_dict["Acc"] + test_result_dict["Precision"] + test_result_dict["Recall"] + test_result_dict["Jaccard"]) / 4
                    lta_keys_02 = ["obs0.2_pred0.1", "obs0.2_pred0.2", "obs0.2_pred0.3", "obs0.2_pred0.5"]
                    lta_keys_03 = ["obs0.3_pred0.1", "obs0.3_pred0.2", "obs0.3_pred0.3", "obs0.3_pred0.5"]
                    lta_keys_05 = ["obs0.5_pred0.1", "obs0.5_pred0.2", "obs0.5_pred0.3", "obs0.5_pred0.5"]
                    lta_f1s_02 = [test_result_dict20.get(k, 0.0) for k in lta_keys_02]
                    lta_f1s_03 = [test_result_dict30.get(k, 0.0) for k in lta_keys_03]
                    lta_f1s_05 = [test_result_dict50.get(k, 0.0) for k in lta_keys_05]
                    lta_score = (sum(lta_f1s_02) + sum(lta_f1s_03) + sum(lta_f1s_05)) / (len(lta_f1s_02) + len(lta_f1s_03) + len(lta_f1s_05))
                    self._save_best_models(tas_score, lta_score, result_dir, tas_metrics=test_result_dict, lta_metrics={'obs0.2': test_result_dict20, 'obs0.3': test_result_dict30, 'obs0.5': test_result_dict50})
                    # =====================

                    w.write(w_)
                    w.close()

                    if result_dir:
                        for k, v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                        np.save(os.path.join(result_dir,
                            f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)


                    if log_train_results:
                        train_result_dict = self.test(
                            train_test_dataset, mode, device, label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k, v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)

                            np.save(os.path.join(result_dir,
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)

                        for k, v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')

        if result_dir:
            logger.close()

        # Print best model performances
        final_summary = "\n" + "="*60 + "\n"
        final_summary += "BEST MODEL PERFORMANCES\n"
        final_summary += "="*60 + "\n"
        final_summary += f"Best TAS Score: {self.best_tas_acc:.2f}\n"
        final_summary += f"Best LTA Score: {self.best_lta_moc:.2f}\n"
        final_summary += f"Best Combined Score: {self.best_both_score:.2f}\n"
        
        if self.best_tas_metrics:
            final_summary += "\nBest TAS Metrics:\n"
            final_summary += f"  - Acc: {self.best_tas_metrics['Acc']:.2f}\n"
            final_summary += f"  - Precision: {self.best_tas_metrics['Precision']:.2f}\n"
            final_summary += f"  - Recall: {self.best_tas_metrics['Recall']:.2f}\n"
            final_summary += f"  - Jaccard: {self.best_tas_metrics['Jaccard']:.2f}\n"
        
        if self.best_lta_metrics:
            final_summary += "\nBest LTA Metrics:\n"
            final_summary += "  obs_p=0.2:\n"
            for key in ["obs0.2_pred0.1", "obs0.2_pred0.2", "obs0.2_pred0.3", "obs0.2_pred0.5"]:
                if key in self.best_lta_metrics['obs0.2']:
                    final_summary += f"    - {key}: {self.best_lta_metrics['obs0.2'][key]:.2f}\n"
            final_summary += "  obs_p=0.3:\n"
            for key in ["obs0.3_pred0.1", "obs0.3_pred0.2", "obs0.3_pred0.3", "obs0.3_pred0.5"]:
                if key in self.best_lta_metrics['obs0.3']:
                    final_summary += f"    - {key}: {self.best_lta_metrics['obs0.3'][key]:.2f}\n"
            final_summary += "  obs_p=0.5:\n"
            for key in ["obs0.5_pred0.1", "obs0.5_pred0.2", "obs0.5_pred0.3", "obs0.5_pred0.5"]:
                if key in self.best_lta_metrics['obs0.5']:
                    final_summary += f"    - {key}: {self.best_lta_metrics['obs0.5'][key]:.2f}\n"
        
        if self.best_combined_metrics:
            final_summary += "\nBest Combined Model Metrics:\n"
            final_summary += "  TAS Metrics:\n"
            tas_metrics = self.best_combined_metrics['tas']
            final_summary += f"    - Acc: {tas_metrics['Acc']:.2f}\n"
            final_summary += f"    - Precision: {tas_metrics['Precision']:.2f}\n"
            final_summary += f"    - Recall: {tas_metrics['Recall']:.2f}\n"
            final_summary += f"    - Jaccard: {tas_metrics['Jaccard']:.2f}\n"
            final_summary += "  LTA Metrics:\n"
            lta_metrics = self.best_combined_metrics['lta']
            final_summary += "    obs_p=0.2:\n"
            for key in ["obs0.2_pred0.1", "obs0.2_pred0.2", "obs0.2_pred0.3", "obs0.2_pred0.5"]:
                if key in lta_metrics['obs0.2']:
                    final_summary += f"      - {key}: {lta_metrics['obs0.2'][key]:.2f}\n"
            final_summary += "    obs_p=0.3:\n"
            for key in ["obs0.3_pred0.1", "obs0.3_pred0.2", "obs0.3_pred0.3", "obs0.3_pred0.5"]:
                if key in lta_metrics['obs0.3']:
                    final_summary += f"      - {key}: {lta_metrics['obs0.3'][key]:.2f}\n"
            final_summary += "    obs_p=0.5:\n"
            for key in ["obs0.5_pred0.1", "obs0.5_pred0.2", "obs0.5_pred0.3", "obs0.5_pred0.5"]:
                if key in lta_metrics['obs0.5']:
                    final_summary += f"      - {key}: {lta_metrics['obs0.5'][key]:.2f}\n"
        
        final_summary += "="*60 + "\n"
        
        # Print to console
        print(final_summary)
        
        # Write to log.txt file
        if result_dir:
            w_path = os.path.join(result_dir, 'log.txt')
            with open(w_path, 'a') as w:
                w.write(final_summary)

    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None, obs_p=0.2):
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None

        with torch.no_grad():
            feature, label, _, video = test_dataset[video_idx]

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]

            input_feats = feature

            # Check if this is anticipation mode (obs_p < 1.0)
            is_anticipation = obs_p < 1.0
            if is_anticipation:
                full_len = feature[0].size(-1)
                obs_len = round(full_len*obs_p)
                input_feats = [feature[i][:,:,:obs_len]
                        for i in range(len(feature))]

            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device))
                       for i in range(len(feature))] # output is a list of tuples
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                if is_anticipation:
                    output = [self.model.ddim_sample(input_feats[i].to(device), seed, full_len=full_len)
                            for i in range(len(input_feats))] # output is a list of tuples
                else:
                    output = [self.model.ddim_sample(feature[i].to(device), seed)
                            for i in range(len(feature))] # output is a list of tuples
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2
                output = [i.cpu() for i in output]

            if mode == 'decoder-noagg':  # temporal aug must be true
                output = [self.model.ddim_sample(feature[len(feature)//2].to(device), seed)] # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0

            assert(output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output = [i[:,:,:min_len] for i in output]
            output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            output = output.mean(0).numpy()

            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            logits = output - np.max(output, axis=0, keepdims=True) 
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=0, keepdims=True)
            hard_labels = np.argmax(output, 0)

            output = restore_full_sequence(hard_labels,
                full_len=label.shape[-1],
                left_offset=left_offset,
                right_offset=right_offset,
                sample_rate=self.sample_rate
            )

            restored_probs = []
            for c in range(probs.shape[0]):
                restored_p = restore_full_sequence(probs[c],
                    full_len=label.shape[-1],
                    left_offset=left_offset,
                    right_offset=right_offset,
                    sample_rate=self.sample_rate
                )
                restored_probs.append(restored_p)
            # 拼接回 [C, Original T] 的形状
            restored_probs = np.stack(restored_probs, axis=0)

            if self.postprocess['type'] == 'mode': # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)

                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:

                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e+1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e-1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e-1]
                            output[mid:ends[e]] = trans[e+1]

            label = label.squeeze(0).cpu().numpy()
            assert(output.shape == label.shape)

            return video, output, restored_probs, label


    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None, all_params=None, obs_p=0.2):

        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        mapping_file = os.path.join(all_params['root_data_dir'], all_params['dataset_name'], 'mapping.txt')
        actions_dict = read_mapping_dict(mapping_file)
        actions_dict_inv = {v: k for k, v in actions_dict.items()}

        # Check if this is anticipation mode (obs_p < 1.0)
        is_anticipation = obs_p < 1.0
        if is_anticipation:
            eval_ps = [0.1, 0.2, 0.3, 0.5]
            # 改为用字典列表记录不同 eval_p 下每个视频独立算出的 MoC
            video_mocs = {i: [] for i in range(len(eval_ps))}

        with torch.no_grad():
            result_dict = {}
            for video_idx in tqdm(range(len(test_dataset))):
                video, pred, probs, label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path, obs_p=obs_p)
                probs = probs.T
                pred_ant = pred
                pred = [self.event_list[int(i)] for i in pred]

                if not os.path.exists(os.path.join(result_dir, 'prediction', str(obs_p))):
                    os.makedirs(os.path.join(result_dir, 'prediction', str(obs_p)))

                file_name = os.path.join(result_dir, 'prediction', str(obs_p), video)
                file_ptr = open(file_name, 'w')
                file_ptr.write('\n'.join(pred))
                file_ptr.close()

                if str(obs_p) == '1.0':
                    if not os.path.exists(os.path.join(result_dir, 'probs')):
                        os.makedirs(os.path.join(result_dir, 'probs'))
                    np.savetxt(os.path.join(result_dir, 'probs', video), probs, fmt='%.6f', delimiter='\t')

                if is_anticipation:
                    total_len = len(label)
                    for i in range(len(eval_ps)):
                        eval_p = eval_ps[i]
                        eval_len = int((obs_p + eval_p) * total_len)
                        eval_pred = pred_ant[:eval_len]
                        T_action, F_action = eval_file(label, eval_pred, obs_p, actions_dict)

                        acc = 0
                        n = 0
                        for j in range(len(actions_dict)):
                            total_actions = T_action[j] + F_action[j]
                            if total_actions != 0:
                                acc += float(T_action[j] / total_actions)
                                n += 1
                        
                        # 如果该视频在当前截断片段内包含有效动作，存入对应的列表
                        if n > 0:
                            single_video_moc = (float(acc) / n) * 100
                            video_mocs[i].append(single_video_moc)

            if is_anticipation:
                for i in range(len(eval_ps)):
                    # 【修改点 3】：取出所有视频的 MoC 进行均值计算 (Macro-average)
                    mocs_list = video_mocs[i]
                    
                    if len(mocs_list) > 0:
                        macro_moc = sum(mocs_list) / len(mocs_list)
                    else:
                        macro_moc = 0.0

                    result = 'obs. %d ' % int(100*obs_p) + 'pred. %d ' % int(100*eval_ps[i]) + '--> Macro MoC: %.2f' % macro_moc
                    result_dict['obs'+str(obs_p)+'_pred'+str(eval_ps[i])] = macro_moc  # Store as percentage
                    print(result)


        if is_anticipation:
            # acc, edit, f1s = func_eval(
            #     label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list, obs_p=obs_p)
            acc_video, precision_phase, recall_phase, jaccard_phase, acc_std, pre_std, recall_std, jaccard_std = eval(label_dir,
                                                                           os.path.join(result_dir, 'prediction', str(obs_p)),
                                                                           test_dataset.video_list, self.event_list, obs_p=obs_p)

        else:
            # acc, edit, f1s = func_eval(
            #     label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list, obs_p=1.0)
            acc_video, precision_phase, recall_phase, jaccard_phase, acc_std, pre_std, recall_std, jaccard_std = eval(label_dir,
                                                                           os.path.join(result_dir, 'prediction', str(obs_p)),
                                                                           test_dataset.video_list, self.event_list,
                                                                           obs_p=1.0)

        result_dict['Acc'] = acc_video
        result_dict['Precision'] = precision_phase
        result_dict['Recall'] = recall_phase
        result_dict['Jaccard'] = jaccard_phase

        if not is_anticipation:
            print("Acc:%.2f"%acc_video, acc_std)
            print("Precision: %.2f"%precision_phase)
            print('Recall: %.2f'%recall_phase)
            print('Jaccard: %.2f'%jaccard_phase, jaccard_std)

        return result_dict

    def _log_tas_results(self, result_dict, text_buffer):
        """Log TAS (Temporal Action Segmentation) results to wandb and text buffer"""
        
        # Add to text buffer
        text_buffer += "TAS Results:\n"
        text_buffer += "Acc: %.2f" % result_dict["Acc"] + '\n'
        text_buffer += "Precision: %.2f" % result_dict["Precision"] + '\n'
        text_buffer += "Recall: %.2f" % result_dict["Recall"] + '\n'
        text_buffer += "Jaccard: %.2f" % result_dict["Jaccard"] + '\n'
        
        return text_buffer

    def _log_lta_results(self, result_dict, text_buffer, obs_p):
        """Log LTA (Long-Term Anticipation) results to wandb and text buffer"""
        
        # Add to text buffer
        text_buffer += f"LTA Results (obs_p={obs_p}):\n"
        text_buffer += f"obs{obs_p}_pred0.1: %.2f" % result_dict[f"obs{obs_p}_pred0.1"] + '\n'
        text_buffer += f"obs{obs_p}_pred0.2: %.2f" % result_dict[f"obs{obs_p}_pred0.2"] + '\n'
        text_buffer += f"obs{obs_p}_pred0.3: %.2f" % result_dict[f"obs{obs_p}_pred0.3"] + '\n'
        text_buffer += f"obs{obs_p}_pred0.5: %.2f" % result_dict[f"obs{obs_p}_pred0.5"] + '\n\n'
        
        return text_buffer

    def _save_best_models(self, tas_acc, lta_moc, result_dir, tas_metrics=None, lta_metrics=None):
        """Save best models based on TAS accuracy, LTA MoC, and combined score"""
        # LTA metric is already scaled to percentage (0.1-0.3 → 10-30)
        # Calculate combined score with equal weights
        combined_score = (tas_acc + lta_moc) / 2
        
        # Check and save best TAS model
        if tas_acc > self.best_tas_acc:
            self.best_tas_acc = tas_acc
            self.best_tas_metrics = tas_metrics
            torch.save(self.model.state_dict(), f'{result_dir}/best_tas_model.pth')
            print(f'New best TAS model saved! Score: {tas_acc:.2f}')
            if tas_metrics:
                print(f'  TAS Metrics - Acc: {tas_metrics["Acc"]:.2f}, Precision: {tas_metrics["Precision"]:.2f}, Recall: {tas_metrics["Recall"]:.2f}, Jaccard: {tas_metrics["Jaccard"]:.2f}')
        
        # Check and save best LTA model
        if lta_moc > self.best_lta_moc:
            self.best_lta_moc = lta_moc
            self.best_lta_metrics = lta_metrics
            torch.save(self.model.state_dict(), f'{result_dir}/best_lta_model.pth')
            print(f'New best LTA model saved! Score: {lta_moc:.2f}')
            if lta_metrics:
                print(f'  LTA Metrics - obs0.2: {lta_metrics["obs0.2"]["obs0.2_pred0.1"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.2"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.3"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.5"]:.2f}, obs0.3: {lta_metrics["obs0.3"]["obs0.3_pred0.1"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.2"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.3"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.5"]:.2f}, obs0.5: {lta_metrics["obs0.5"]["obs0.5_pred0.1"]:.2f}/{lta_metrics["obs0.5"]["obs0.5_pred0.2"]:.2f}/{lta_metrics["obs0.5"]["obs0.5_pred0.3"]:.2f}/{lta_metrics["obs0.5"]["obs0.5_pred0.5"]:.2f}')
        
        # Check and save best combined model
        if combined_score > self.best_both_score:
            self.best_both_score = combined_score
            self.best_combined_metrics = {'tas': tas_metrics, 'lta': lta_metrics}
            torch.save(self.model.state_dict(), f'{result_dir}/best_combined_model.pth')
            print(f'New best combined model saved! Score: {combined_score:.2f} (TAS: {tas_acc:.2f}, LTA: {lta_moc:.2f})')
            if tas_metrics and lta_metrics:
                print(f'  TAS Metrics - Acc: {tas_metrics["Acc"]:.2f}, Precision: {tas_metrics["Precision"]:.2f}, Recall: {tas_metrics["Recall"]:.2f}, Jaccard: {tas_metrics["Jaccard"]:.2f}')
                print(f'  LTA Metrics - obs0.2: {lta_metrics["obs0.2"]["obs0.2_pred0.1"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.2"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.3"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.5"]:.2f}, obs0.3: {lta_metrics["obs0.3"]["obs0.3_pred0.1"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.2"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.3"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.5"]:.2f}, obs0.5: {lta_metrics["obs0.5"]["obs0.5_pred0.1"]:.2f}/{lta_metrics["obs0.5"]["obs0.5_pred0.2"]:.2f}/{lta_metrics["obs0.5"]["obs0.5_pred0.3"]:.2f}/{lta_metrics["obs0.5"]["obs0.5_pred0.5"]:.2f}')