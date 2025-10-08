import os
import torch
import torch.nn as nn
import numpy as np

from os.path import join
from tqdm import tqdm
from collections import defaultdict

from torch.optim.lr_scheduler import StepLR


class GeneralTrainer:
    def __init__(self, model, train_config, pixel_loss=None, regularizer=None, device=None, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = model.to(self.device)
        self.train_config = train_config
        self.num_epochs = train_config.get('num_epochs', 200)
        self.verbose = train_config.get('verbose', 1)
        self.learning_rate = train_config.get('learning_rate', 1e-4)
        self.step_size = train_config.get('step_size', 100)
        self.gamma = train_config.get('gamma', 0.95)
        self.weight_decay = train_config.get('weight_decay', 0.0)
        self.gradient_clip = train_config.get('gradient_clip', False)
        self.gradient_clip_val = train_config.get('gradient_clip_val', None)
        self.kwargs = kwargs

        # Loss function
        self.loss_fn = nn.MSELoss() if pixel_loss is None else pixel_loss
        self.regularizer = regularizer
        self.regularizer_weight = train_config.get('regularizer_weight', None)

        # Optimizer and LR scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def forward(self, data, return_tensors=True, is_training=True, epoch=None):
        """
        Must be implemented by subclass. Should return:
        {
            'tensors': {...},    # any tensor-like outputs
            'losses': {...}      # all loss terms, including 'loss', 'loss_pixel', etc.
        }
        """
        raise NotImplementedError

    def load_checkpoint(self, path_to_ckpt):
        
        # Load checkpoint
        checkpoint = torch.load(join(path_to_ckpt, 'checkpoint.pt'), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load loss history
        start_epoch = 0
        loss_tracker_dict = defaultdict(list)
        learning_rate_list = []
        loss_file = join(path_to_ckpt, 'loss_epoch.npz')
        loss_np_dict = np.load(loss_file)

        for k in loss_np_dict.files:
            if k == 'learning_rate':
                learning_rate_list = list(loss_np_dict[k])
            else:
                loss_tracker_dict[k] = list(loss_np_dict[k])
        
        start_epoch = len(learning_rate_list)
            
        return loss_tracker_dict, learning_rate_list, start_epoch

    def save_checkpoint(self, epoch, path_to_ckpt, loss_tracker_dict, learning_rate_list, save_epoch_ckpt=True):

        ckpt_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        if save_epoch_ckpt:
            torch.save(ckpt_data, join(path_to_ckpt, f'checkpoint{epoch}.pt'))

        torch.save(ckpt_data, join(path_to_ckpt, 'checkpoint.pt'))

        # Save losses
        loss_np_dict = {k: np.array(v) for k, v in loss_tracker_dict.items()}
        loss_np_dict['learning_rate'] = np.array(learning_rate_list)
        np.savez(join(path_to_ckpt, 'loss_epoch.npz'), **loss_np_dict)

    def _run_one_epoch(self, dataloader, epoch=None, train=True):
        mode = 'train' if train else 'valid'
        self.model.train() if train else self.model.eval()
        loop = tqdm(dataloader, desc=f"{mode.capitalize()}ing") if self.verbose else dataloader

        running_losses = defaultdict(float)
        num_batches = 0

        for data in loop:
            if train:
                self.optimizer.zero_grad()
                output = self.forward(data, return_tensors=False, is_training=True, epoch=epoch)
                losses = output['losses']
                loss = losses['loss']  # main loss
                loss.backward()

                # Clip gradients if specified
                if self.gradient_clip and self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                # Update optimizer and scheduler
                self.optimizer.step()
                self.scheduler.step()

            else:
                with torch.no_grad():
                    output = self.forward(data, return_tensors=False, is_training=False, epoch=epoch)
                    losses = output['losses']                

            # Update statistics
            for k, v in losses.items():
                running_losses[k] += v.item()
            num_batches += 1

            if self.verbose:
                loop.set_postfix({k: f"{(v / num_batches):.4f}" for k, v in running_losses.items()})

        avg_losses = {k: v / num_batches for k, v in running_losses.items()}
        return avg_losses

    def _format_losses(self, loss_dict, width=6):
        formatted_losses = []
        for k, v in loss_dict.items():
            key_width = max(len(k), width) 
            formatted_losses.append(f"{k.ljust(key_width)}: {v:.4f}")
        return ' | '.join(formatted_losses)
    
    def train(self, train_loader, valid_loader, path_to_model, ckpt_epoch=5, resume=False, resume_path=None):
        # Resume training if specified
        if resume and resume_path is not None:
            loss_tracker_dict, learning_rate_list, start_epoch = self.load_checkpoint(resume_path)
        else:
            loss_tracker_dict = defaultdict(list)
            learning_rate_list = []
            start_epoch = 0

        for epoch in range(start_epoch, self.num_epochs):
            # Train and validate
            train_losses = self._run_one_epoch(train_loader, epoch, train=True)
            valid_losses = self._run_one_epoch(valid_loader, epoch, train=False)

            # Store losses and learning rates
            for mode, losses in zip(['train', 'valid'], [train_losses, valid_losses]):
                for k, v in losses.items():
                    key = f'{mode}_{k}'
                    if key not in loss_tracker_dict:
                        loss_tracker_dict[key] = []
                    loss_tracker_dict[key].append(v)
            learning_rate_list.append(self.optimizer.param_groups[0]['lr'])

            # Print losses
            if self.verbose:
                train_loss_str = self._format_losses(train_losses)
                valid_loss_str = self._format_losses(valid_losses)
                print(f"{'Epoch':<6} {epoch+1:03d}")
                print(f"{'Train Losses':<14} | {train_loss_str}")
                print(f"{'Valid Losses':<14} | {valid_loss_str}")

            # Save checkpoint
            self.save_checkpoint(
                epoch=epoch,
                path_to_ckpt=path_to_model,
                loss_tracker_dict=loss_tracker_dict,
                learning_rate_list=learning_rate_list,
                save_epoch_ckpt=((epoch + 1) % ckpt_epoch == 0)
            )

        return loss_tracker_dict

    @torch.inference_mode()
    def test(self, test_loader):
        self.model.eval()
        all_outputs = defaultdict(list)
        running_losses = defaultdict(float)
        num_batches = 0

        loop = tqdm(test_loader, desc="Testing") if self.verbose else test_loader

        for data in loop:
            output = self.forward(data, return_tensors=True, is_training=False)

            for k, v in output['losses'].items():
                running_losses[k] += v.item()
            for k, v in output['tensors'].items():
                all_outputs[k].append(v.cpu())

            num_batches += 1
            if self.verbose:
                loop.set_postfix({k: f"{v / num_batches:.4f}" for k, v in running_losses.items()})

        averaged_losses = {k: v / num_batches for k, v in running_losses.items()}
        output_tensors = {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}

        return {
            'losses': averaged_losses,
            'tensors': output_tensors
        }

