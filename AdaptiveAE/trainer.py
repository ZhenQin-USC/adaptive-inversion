import os
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from os.path import join
from torch.optim.lr_scheduler import StepLR
from .nets import AdaptiveAutoEncoder
from General import PatchDiscriminator
from General import PatchAdversarialLoss
from General import PerceptualLoss


class AdaAETrainer:
    def __init__(self, model_config, train_config, Model=AdaptiveAutoEncoder, **kwargs):
        super().__init__()

        device = kwargs.get('device', None)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.train_config = train_config
        self.kwargs = kwargs

        # training configuration
        self.num_epoch = train_config.get('num_epoch')
        self.batch_size = train_config.get('batch_size')
        self.verbose = train_config.get('verbose')
        self.gradient_clip = train_config.get('gradient_clip')
        self.gradient_clip_val = train_config.get('gradient_clip_val')
        self.use_amp = train_config.get('use_amp', False)
        self.loss_mode = train_config.get('loss_mode', 'L2').upper()  # 'L1', 'L2 (MSE)' 'HYBRID'
        self.learning_rate = train_config.get('learning_rate')
        self.weight_decay = float(train_config.get('weight_decay', 0.0))
        # Build Model
        self._model = Model(units=model_config.get('units'), model_config=model_config).to(self.device)

        # Define Loss Function
        if self.loss_mode == 'L2' or self.loss_mode == 'MSE':
            self._loss_fn = nn.MSELoss() 
        elif self.loss_mode == 'L1':
            self._loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Only support l1 or l2 (mse), doesn't support {self.loss_mode}")

        # Define Optimizer
        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=train_config.get('learning_rate'),
                                           weight_decay=self.weight_decay)
        self._scheduler = StepLR(self._optimizer,
                                 step_size=train_config.get('step_size'),
                                 gamma=train_config.get('gamma'),
                                 verbose=False) # 'deprecated'
        
    def _weight_initialization(self, m):
        # custom weights initialization for both networks
        classname = m.__class__.__name__

        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def _ensure_directory_exists(self, path_to_model: str):
        """
        Check if the directory exists; if not, create it.

        Parameters:
        - path_to_model: The path to the directory to check/create.
        """
        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)
            print(f"Directory created: {path_to_model}")
        else:
            print(f"Directory already exists: {path_to_model}")

    def _loss(self, recons, target):
        for idx, recon in enumerate(recons):
            if idx == 0:
                loss = self._loss_fn(recon, target)
                loss_list = [loss.item()]
            else:
                _loss = self._loss_fn(recon, target)
                loss_list.append(_loss.item())
                loss += _loss
        return loss, loss_list

    def _print_loss(self, epoch, batch_loss, batch_vloss, all_losses, all_vlosses):
        # Print loss
        loss_list, vloss_list = '', ''
        for idx, _ in enumerate(all_losses):
            loss_list += ' Level {}: {:.6f} |'.format(idx+1, _)
        for idx, _ in enumerate(all_vlosses):
            vloss_list += ' Level {}: {:.6f} |'.format(idx+1, _)
        print('Epoch {:5s}: Train loss -- total {:.6f}  |'.format(str(epoch), batch_loss) + loss_list)
        print('             Valid loss -- total {:.6f}  |'.format(batch_vloss) + vloss_list)
        print('         LR: {}'.format(self._scheduler.get_last_lr()))

    def _load_model(self, path_to_ckpt):
        checkpoint_path = join(path_to_ckpt, 'checkpoint.pt')
        if torch.cuda.is_available() and self.device == torch.device("cuda"):
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load_checkpoint(self, path_to_model):
        """Load model and optimizer states along with loss data."""
        self._load_model(path_to_model)

        loss_data = np.load(join(path_to_model, 'loss_epoch.npz'))
        train_loss_list = list(loss_data['train_loss'])
        valid_loss_list = list(loss_data['valid_loss'])
        learning_rate_list = list(loss_data['learning_rate'])

        return train_loss_list, valid_loss_list, learning_rate_list
    
    def _train_one_epoch(self, train_loader, epoch):

        batch_loss = 0.0
        losses_all_levels = []

        if self.verbose == 1:
            loop = tqdm(train_loader)
        elif self.verbose == 0:
            loop = train_loader

        self._model.train()

        for i, data in enumerate(loop):
            _m = data.to(self.device)
            self._optimizer.zero_grad()

            out = self.forward(_m)
            loss = out['loss']
            loss.backward()

            if self.gradient_clip and self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.gradient_clip_val)

            # Updates the parameters and adjust learning weights
            self._optimizer.step()
            self._scheduler.step()

            # Gather data and report
            batch_loss += loss.item()
            if len(losses_all_levels) == 0:
                losses_all_levels = [0.0]*len(out['loss_list'])
            for idx, loss_item in enumerate(out['loss_list']):
                losses_all_levels[idx] += loss_item

            # Print loss log
            if self.verbose == 1:
                loss_list:list = out['loss_list']
                bar_info = f' Mixed: {loss_list[0]:.6f}'
                for idx, _ in enumerate(loss_list[1:]):
                    bar_info += '| Level {}: {:.6f} '.format(idx+1, _)
                loop.set_description(f"Epoch [{epoch}/{self.num_epoch}]")
                loop.set_postfix(
                    loss=loss.item(), 
                    info=bar_info,
                                )
            elif self.verbose == 0:
                pass
        
        return batch_loss/(i+1), [_/(i+1) for _ in losses_all_levels]

    def _validate(self, valid_loader):

        running_vloss = 0.0
        losses_all_levels = []

        self._model.eval()
        for step, data in enumerate(valid_loader):

            _m = data.to(self.device)
            with torch.no_grad():
                out = self.forward(_m)
                vloss = out['loss']

            running_vloss += vloss.item()
            if len(losses_all_levels) == 0:
                losses_all_levels = [0.0]*len(out['loss_list'])
            for idx, loss_item in enumerate(out['loss_list']):
                losses_all_levels[idx] += loss_item

        return running_vloss/(step+1), [_/(step+1) for _ in losses_all_levels]

    def forward(self, x, indices=None):

        recons, indices, indices_repeat = self._model(x, indices)
        loss, loss_list = self._loss(recons, x)

        return {'loss': loss, 
                'recon': torch.stack(recons, dim=1), 
                'loss_list': loss_list, 
                'indices': indices, 
                'indices_repeat': indices_repeat
                }

    def train(self, 
              path_to_model, 
              train_loader, 
              valid_loader, 
              test_loader=None,
              if_track_validate=True, 
              ckpt_epoch:int=5,
              plot_test_recon:bool=False,
              start_epoch:int=0
              ):
        
        # Create the directory if it doesn't exist
        self._ensure_directory_exists(path_to_model)

        train_loss_list, valid_loss_list, learning_rate_list = (
            self.load_checkpoint(path_to_model) if start_epoch > 0 else ([], [], [])
            )   # load checkpoint if it starts from a specified epoch else just empty lists

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(start_epoch, self.num_epoch):  # Start from start_epoch
            # train one epoch
            batch_loss, all_losses = self._train_one_epoch(train_loader, epoch)

            # store the losses
            train_loss_list.append(batch_loss)
            learning_rate_list.append(float(self._optimizer.param_groups[0]['lr']))

            # validate
            if if_track_validate == True:
                batch_vloss, all_vlosses = self._validate(valid_loader)
            else:
                batch_vloss, all_vlosses = 0.0, [0.0]
            valid_loss_list.append(batch_vloss)

            # test
            if plot_test_recon and test_loader is not None:
                test_true, test_recon, test_loss, loss_list_collect = self.test(test_loader)
                self.plot_recons(test_recon, test_true, join(path_to_model, f"test_recon_epoch_{epoch}.png"))
                self.plot_hists(loss_list_collect, join(path_to_model, f"test_distribution_epoch_{epoch}.png"))

            # Print loss
            self._print_loss(epoch, batch_loss, batch_vloss, all_losses, all_vlosses)

            # Save checkpoint every ckpt_epoch epochs
            if (epoch+1)%ckpt_epoch == 0:
                torch.save({
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    },
                    join(path_to_model, 'checkpoint{}.pt'.format(epoch)))

            # Save checkpoint every epoch
            torch.save({
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
                },
                join(path_to_model, 'checkpoint.pt'))

            # save losses
            np.savez(join(path_to_model, 'loss_epoch.npz'),
                     train_loss=np.array(train_loss_list),
                     valid_loss=np.array(valid_loss_list),
                     learning_rate=np.array(learning_rate_list)
                     )

        return train_loss_list, valid_loss_list

    def test(self, test_loader):
        i, running_loss = 0, 0.0
        outputs = []
        trues = []
        loss_list_collect = []
        self._model.eval()

        if self.verbose == 1:
            loop = tqdm(test_loader)
        elif self.verbose == 0:
            loop = test_loader

        for i, data in enumerate(loop):
            _m = data.to(self.device)
            with torch.no_grad():
                out = self.forward(_m)
                vloss = out['loss']
                loss_list = out['loss_list']
            
            loss_list_collect.append(loss_list)
            running_loss += vloss.item()
            outputs.append(out['recon'].detach().cpu())
            trues.append(_m.detach().cpu())

            if self.verbose == 1:
                loop.set_description("Test bar: ")
                loop.set_postfix(loss=running_loss/(i+1))

        return torch.vstack(trues), torch.vstack(outputs), running_loss/(i+1), np.array(loss_list_collect).T

    def plot_masked_recons(self, test_recon, test_mixed, test_true, indices, mask_type='transparent',
                           figure_settings=None, aspect=15, shrink=0.9, pad=0.01):
        # plot
        # aspect, shrink = 15, 0.90
        nrows = test_recon.shape[1]+3
        if figure_settings == None:
            ncols, dpi = 10, 400
            figure_settings = {'nrows': nrows, 'ncols': ncols, 'figsize': (ncols, nrows), 'dpi': dpi}

        fig, axs = plt.subplots(**figure_settings)

        for i in range(ncols):
            im_true = axs[0,i].imshow(test_true[i, 0, :, :, 0].detach().cpu().numpy(), vmin=0, vmax=1.0, cmap='jet')

            for j in range(test_recon.shape[1]):
                if mask_type == 'transparent':
                    axs[j+1,i].imshow(test_recon[i, j, 0, :, :, 0].detach().cpu().numpy(), vmin=0, vmax=1.0, cmap='jet')
                    mask = np.zeros_like(test_recon[i, j, 0, :, :, 0].detach().cpu().numpy())
                    mask[indices[i] == j] = 1.0 # Assuming indices[i] contains row indices for masking
                    axs[j + 1, i].imshow(mask, cmap='gray', alpha=0.2, vmin=0, vmax=1)
                elif mask_type == 'mask':
                    recon_image = test_recon[i, j, 0, :, :, 0].detach().cpu().numpy()
                    mask = indices[i] != j  # Create a mask where indices match the current row
                    # Create an RGBA image based on the recon_image
                    rgba_image = np.zeros((recon_image.shape[0], recon_image.shape[1], 4))
                    rgba_image[..., :3] = plt.cm.jet(recon_image)[..., :3]  # Use the colormap to set RGB
                    rgba_image[..., 3] = np.where(mask, 0, 1)  # Set alpha to 0 where mask is True, else 1
                    axs[j + 1, i].imshow(rgba_image)  # Display the RGBA image
                else:
                    raise ValueError("Invalid mask_type. Must be 'transparent' or 'mask'.")

            axs[-2,i].imshow(test_mixed[i,0,:,:,0].detach().cpu().numpy(), vmin=0, vmax=1.0, cmap='jet')
            error = test_mixed[i,0,:,:,0].detach().cpu().numpy() - test_true[i,0,:,:,0].detach().cpu().numpy()
            im_error = axs[-1,i].imshow(error, vmin=-0.2, vmax=0.2, cmap='seismic')

            for j in range(nrows):
                axs[j,i].set_xticks([]), axs[j,i].set_yticks([])

        axs[0,0].set_ylabel('True', fontsize=10)

        for j in range(1,test_recon.shape[1]+1):
            axs[j,0].set_ylabel('Level {}'.format(j), fontsize=10)
        axs[-2,0].set_ylabel('Mixed', fontsize=10)
        axs[-1,0].set_ylabel('Error', fontsize=10)

        fig.colorbar(im_true, ax=axs[0:-1, :], shrink=shrink, pad=pad, aspect=aspect*(nrows-1))
        fig.colorbar(im_error, ax=axs[-1, :], shrink=shrink, pad=pad, aspect=aspect)

        plt.show()

    def compare_recons(self, test_recon1, test_recon2, mask=None, figure_settings=None, aspect=15, shrink=0.9, pad=0.01):
        nrows = 3
        if figure_settings == None:
            ncols, dpi = 10, 400
            figure_settings = {'nrows': nrows, 'ncols': ncols, 'figsize': (ncols, nrows), 'dpi': dpi}

        fig, axs = plt.subplots(**figure_settings)

        for i in range(ncols):
            im1 = axs[0,i].imshow(test_recon1[i,0,:,:,0].detach().cpu().numpy(), vmin=0, vmax=1.0, cmap='jet')
            im2 = axs[1,i].imshow(test_recon2[i,0,:,:,0].detach().cpu().numpy(), vmin=0, vmax=1.0, cmap='jet')
            error = test_recon1[i,0,:,:,0].detach().cpu().numpy() - test_recon2[i,0,:,:,0].detach().cpu().numpy()
            im_error = axs[-1,i].imshow(error, vmin=-0.2, vmax=0.2, cmap='seismic')

            if mask is not None:
                axs[1, i].imshow(mask[i], cmap='gray', alpha=0.2, vmin=0, vmax=1)

            for j in range(nrows):
                axs[j,i].set_xticks([]), axs[j, i].set_yticks([])

        axs[0,0].set_ylabel('Recon1', fontsize=10)
        axs[1,0].set_ylabel('Recon2', fontsize=10)
        axs[-1,0].set_ylabel('Error', fontsize=10)

        fig.colorbar(im1, ax=axs[0:-1, :], shrink=shrink, pad=pad, aspect=aspect*(nrows-1))
        fig.colorbar(im_error, ax=axs[-1, :], shrink=shrink, pad=pad, aspect=aspect)

        plt.show()

    def plot_recons(self, test_recon, test_true, save_path=None, num_cases=10, figure_settings=None):
        nrows = test_recon.shape[1]+2
        img_height, img_width = test_true.shape[2], test_true.shape[3]
        aspect_ratio = img_width / img_height

        if figure_settings == None:
            ncols = num_cases
            dpi = 400
            base_height = 1  # Base height for each row
            fig_width = ncols * base_height * aspect_ratio
            fig_height = nrows * base_height
            figure_settings = {
                'nrows': nrows,
                'ncols': ncols,
                'figsize': (fig_width, fig_height),
                'dpi': dpi,
            }

        fig, axs = plt.subplots(**figure_settings)
        for i in range(num_cases):
            axs[0,i].imshow(test_true[i,0,:,:,0].detach().cpu().numpy(), vmin=0, vmax=1.0, cmap='jet')

            for j in range(test_recon.shape[1]):
                axs[j+1,i].imshow(test_recon[i,j,0,:,:,0].detach().cpu().numpy(), vmin=0, vmax=1.0, cmap='jet')
            error = test_recon[i,-1,0,:,:,0].detach().cpu().numpy() - test_true[i,0,:,:,0].detach().cpu().numpy()
            axs[-1,i].imshow(error, vmin=-0.2, vmax=0.2, cmap='seismic')

            for j in range(nrows):
                axs[j,i].set_xticks([]), axs[j,i].set_yticks([])

        axs[0,0].set_ylabel('True', fontsize=10)
        axs[1,0].set_ylabel('Mixed', fontsize=10)
        for j in range(2, test_recon.shape[1]+1):
            axs[j,0].set_ylabel('Level {}'.format(j-1), fontsize=10)
        axs[-1,0].set_ylabel('Error', fontsize=10)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)  
            
    def plot_hists(self, loss_list_collect, save_path, figure_settings=None, bin_settings=None):
        
        if bin_settings is None:
            bin_settings = {'bins': 30, 'histtype': 'stepfilled'}
        
        if figure_settings is None:    
            figure_settings = {'figsize': (6, 3), 'dpi': 200}
        
        plt.figure(**figure_settings)
        
        colors = ['gray', 'skyblue', 'salmon', 'lightgreen', 'gold', 
                  'orchid', 'royalblue', 'darkorange', 'teal', 'mediumseagreen']
        edge_color = 'black' 
        
        for i, loss_array in enumerate(loss_list_collect):
            label = 'Mixed' if i == 0 else f'Level {i}'
            plt.hist(loss_array, **bin_settings, color=colors[i % len(colors)], 
                    edgecolor=edge_color, alpha=0.75, label=label)
        
        plt.legend(title='Scales')
        plt.title('Distributions of Test Errors')
        plt.xlabel('RMSE Values')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close() 
            

class AdversarialAdaAETrainer(AdaAETrainer):
    def __init__(self, model_config, disc_config, train_config, Model=AdaptiveAutoEncoder, **kwargs):
        super().__init__(model_config, train_config, Model=Model, **kwargs)

        self._perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg").to(self.device)
        self._discriminator = PatchDiscriminator(**disc_config).to(self.device)
        self._adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.warmup_epochs = train_config.get('autoencoder_warm_up_n_epochs')

        self.adv_weight = train_config.get('adv_weight', 1.0)
        self.perceptual_weight = train_config.get('perceptual_weight')

        self._optimizer_d = torch.optim.Adam(params=self._discriminator.parameters(), lr=self.learning_rate)
        
        self.perceptual_level = train_config.get('perceptual_level', 1)
        self.rng = np.random.default_rng(train_config.get('seed', 42))

    def perceptual_loss(self, _m, recons):
        _recon = recons[:,0] # [y_mix, y_1, ..., y_N]
        p_loss = self._perceptual_loss(_m.permute((0, -1, 1, 2, 3)).contiguous().view(-1, 1, _m.size(2), _m.size(3)), 
                                        _recon.permute((0, -1, 1, 2, 3)).contiguous().view(-1, 1, _recon.size(2), _recon.size(3)))
        
        scale_indices = list(range(recons.size(1)))[1:] # ignore the first one
        shuffled_indices = self.rng.permutation(scale_indices)
        
        for scale_idx in shuffled_indices[:self.perceptual_level]:
            _recon = recons[:, scale_idx] 
            p_loss += self._perceptual_loss(_m.permute((0, -1, 1, 2, 3)).contiguous().view(-1, 1, _m.size(2), _m.size(3)), 
                                            _recon.permute((0, -1, 1, 2, 3)).contiguous().view(-1, 1, _recon.size(2), _recon.size(3)))

        return p_loss

    def _train_one_epoch(self, train_loader, epoch):

        batch_loss, gen_epoch_loss, disc_epoch_loss, percept_epoch_loss = 0.0, 0.0, 0.0, 0.0
        losses_all_levels = []

        if self.verbose == 1:
            loop = tqdm(train_loader)
        elif self.verbose == 0:
            loop = train_loader

        self._model.train()
        self._discriminator.train()

        for step, data in enumerate(loop):
            _m = data.to(self.device)

            # Generator part
            self._optimizer.zero_grad()

            out = self.forward(_m)
            loss = out['loss'] 
            recons = out['recon']
            p_loss = self.perceptual_loss(_m, recons)
            
            if self.perceptual_weight is not None:
                loss += self.perceptual_weight * p_loss

            if epoch >= self.warmup_epochs:
                logits_fake = self._discriminator(recons[:,-1])[-1]
                generator_loss = self._adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss += self.adv_weight * generator_loss
            
            loss.backward()

            if self.gradient_clip and self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.gradient_clip_val)
            
            # Updates the parameters and adjust learning weights
            self._optimizer.step()

            if epoch >= self.warmup_epochs:
                self._optimizer_d.zero_grad()
                logits_fake = self._discriminator(recons[:,-1].contiguous().detach())[-1]
                loss_d_fake = self._adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = self._discriminator(_m.contiguous().detach())[-1]
                loss_d_real = self._adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = self.adv_weight * discriminator_loss
                
                loss_d.backward()
                self._optimizer_d.step()

            self._scheduler.step()

            # Gather data and report
            batch_loss += loss.item()
            percept_epoch_loss += p_loss.item()

            if epoch >= self.warmup_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            if len(losses_all_levels) == 0:
                losses_all_levels = [0.0]*len(out['loss_list'])
            
            for idx, loss_item in enumerate(out['loss_list']):
                losses_all_levels[idx] += loss_item
                    
            loss_list:list = out['loss_list']
            bar_info = f" Mixed: {loss_list[0]:.6f}"
            
            for idx, val in enumerate(loss_list[1:]):
                bar_info += f" | Level {idx+1}: {val:.6f}"

            bar_info += f" | Perceptual: {p_loss.item():.6f}"

            loss_dict = {
                "batch_loss": batch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
                "perceptual": percept_epoch_loss / (step + 1), 
                "info": bar_info
            }
            
            # Print loss log
            if self.verbose == 1:
                loop.set_description(f"Epoch [{epoch}/{self.num_epoch}]")
                loop.set_postfix(loss=loss.item(), info=bar_info)
            elif self.verbose == 0:
                pass

            loss_dict['all_levels_loss'] = [_/(step+1) for _ in losses_all_levels]
        return loss_dict

    def _save_model(self, path_to_ckpt):
        torch.save(
            {
                'autoencoder_state_dict': self._model.state_dict(),
                'discriminator_state_dict': self._discriminator.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
                'optimizer_d_state_dict': self._optimizer_d.state_dict()
            }, path_to_ckpt)

    def _load_model(self, path_to_ckpt):
        checkpoint_path = join(path_to_ckpt, 'checkpoint.pt')
        if torch.cuda.is_available() and self.device == torch.device("cuda"):
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        self._model.load_state_dict(checkpoint['autoencoder_state_dict'])
        self._discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    def load_checkpoint(self, path_to_model):
        """Load model and optimizer states along with loss data."""
        self._load_model(path_to_model)

        loss_data = np.load(join(path_to_model, 'loss_epoch.npz'))
        train_loss_list = list(loss_data['train_loss'])
        valid_loss_list = list(loss_data['valid_loss'])
        learning_rate_list = list(loss_data['learning_rate'])
        epoch_gen_loss_list = list(loss_data["gen_loss_list"])
        epoch_disc_loss_list = list(loss_data["disc_loss_list"])
        epoch_perceptual_list = list(loss_data["perceptual_list"])

        return train_loss_list, valid_loss_list, learning_rate_list, epoch_gen_loss_list, epoch_disc_loss_list, epoch_perceptual_list

    def _print_loss(self, epoch, loss_dict, batch_vloss, all_vlosses):
        all_levels_loss = loss_dict['all_levels_loss']
        loss_info, vloss_info = f' Mixed: {all_levels_loss[0]:.6f}', f' Mixed: {all_vlosses[0]:.6f}'
        for idx, val in enumerate(all_levels_loss[1:]):
            loss_info += f" | Level {idx+1}: {val:.6f}"
        for idx, val in enumerate(all_vlosses[1:]):
            vloss_info += f" | Level {idx+1}: {val:.6f}"
        
        loss_info += ' | G loss: {:.6f}'.format(loss_dict['gen_loss'])
        loss_info += ' | D loss: {:.6f}'.format(loss_dict['disc_loss'])
        loss_info += ' | P loss: {:.6f}'.format(loss_dict['perceptual'])

        print("Epoch {:5s}: Train loss -- total {:.6f} |".format(str(epoch), loss_dict['batch_loss']) + loss_info)
        print("             Valid loss -- total {:.6f} |".format(batch_vloss) + vloss_info)
        print('         LR: {}'.format(self._scheduler.get_last_lr()))

    def train(self, 
              path_to_model, 
              train_loader, 
              valid_loader, 
              test_loader=None,
              if_track_validate=True, 
              ckpt_epoch:int=5,
              plot_test_recon:bool=False,
              start_epoch:int=0
              ):
        
        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)

        train_loss_list, valid_loss_list, learning_rate_list, \
            epoch_gen_loss_list, epoch_disc_loss_list, epoch_perceptual_list = (
            self.load_checkpoint(path_to_model) if start_epoch > 0 else ([], [], [], [], [], [])
            )   # load checkpoint if it starts from a specified epoch else just empty lists

        for epoch in range(start_epoch, self.num_epoch):  # Start from start_epoch
            
            # train one epoch
            _loss_dict = self._train_one_epoch(train_loader, epoch)

            # store the losses
            train_loss_list.append(_loss_dict['batch_loss'])
            epoch_gen_loss_list.append(_loss_dict['gen_loss'])
            epoch_disc_loss_list.append(_loss_dict['disc_loss'])
            epoch_perceptual_list.append(_loss_dict['perceptual'])
            learning_rate_list.append(float(self._optimizer.param_groups[0]['lr']))

            # validate
            if if_track_validate == True:
                batch_vloss, all_vlosses = self._validate(valid_loader)
            else:
                batch_vloss, all_vlosses = 0.0, [0.0]
            valid_loss_list.append(batch_vloss)
            
            # test
            if plot_test_recon and test_loader is not None:
                test_true, test_recon, test_loss, loss_list_collect = self.test(test_loader)
                self.plot_recons(test_recon, test_true, join(path_to_model, f"test_recon_epoch_{epoch}.png"))
                self.plot_hists(loss_list_collect, join(path_to_model, f"test_distribution_epoch_{epoch}.png"))

            # print losses 
            self._print_loss(epoch, _loss_dict, batch_vloss, all_vlosses)

            # Save checkpoint every ckpt_epoch epochs
            if (epoch+1)%ckpt_epoch == 0:
                self._save_model(join(path_to_model, 'checkpoint{}.pt'.format(epoch)))
            
            # Save checkpoint every epoch
            self._save_model(join(path_to_model, 'checkpoint.pt'))

            # save losses
            np.savez(join(path_to_model, 'loss_epoch.npz'),
                     train_loss=np.array(train_loss_list),
                     valid_loss=np.array(valid_loss_list),
                     gen_loss_list=np.array(epoch_gen_loss_list), 
                     disc_loss_list=np.array(epoch_disc_loss_list), 
                     perceptual_list=np.array(epoch_perceptual_list),
                     learning_rate=np.array(learning_rate_list)
                     )

        return train_loss_list, valid_loss_list


class AdversarialAdaVAETrainer(AdaAETrainer):
    def __init__(self, model_config, disc_config, train_config, Model, **kwargs):
        super().__init__(model_config, train_config, Model=Model, **kwargs)

        self._perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg").to(self.device)
        self._discriminator = PatchDiscriminator(**disc_config).to(self.device)
        self._adv_loss = PatchAdversarialLoss(criterion="least_squares")

        self.adv_weight = train_config.get('adv_weight', 1.0)
        self.kld_weight = train_config.get('kld_weight')
        self.perceptual_weight = train_config.get('perceptual_weight')
        self.warmup_epochs = train_config.get('autoencoder_warm_up_n_epochs')
        self._optimizer_d = torch.optim.Adam(params=self._discriminator.parameters(), 
                                             lr=self.learning_rate)
        
        self.perceptual_level = train_config.get('perceptual_level', 1)
        self.rng = np.random.default_rng(seed=42)  # seed is optional

    def forward(self, x, indices=None):

        recons, indices, indices_repeat, mu, logvar, kl_loss = self._model(x, indices)
        loss, loss_list = self._loss(recons, x)

        return {'loss': loss, 
                'kl_loss': kl_loss,
                'recon': torch.stack(recons, dim=1), 
                'loss_list': loss_list, 
                'indices': indices, 
                'indices_repeat': indices_repeat
                }

    def perceptual_loss(self, _m, recons):
        _recon = recons[:,0] # [y_mix, y_1, ..., y_N]
        p_loss = self._perceptual_loss(_m.permute((0, -1, 1, 2, 3)).contiguous().view(-1, 1, _m.size(2), _m.size(3)), 
                                        _recon.permute((0, -1, 1, 2, 3)).contiguous().view(-1, 1, _recon.size(2), _recon.size(3)))
        
        scale_indices = list(range(recons.size(1)))[1:] # ignore the first one
        shuffled_indices = self.rng.permutation(scale_indices)
        
        for scale_idx in shuffled_indices[:self.perceptual_level]:
            _recon = recons[:, scale_idx] 
            p_loss += self._perceptual_loss(_m.permute((0, -1, 1, 2, 3)).contiguous().view(-1, 1, _m.size(2), _m.size(3)), 
                                            _recon.permute((0, -1, 1, 2, 3)).contiguous().view(-1, 1, _recon.size(2), _recon.size(3)))

        return p_loss

    def kl_divergence_loss(self, kl_loss_list):
        return sum(kl_loss_list)
    
    def _loss(self, recons, target):
        for idx, recon in enumerate(recons):
            if idx == 0:
                loss = self._loss_fn(recon, target)
                loss_list = [loss.item()]
            else:
                _loss = self._loss_fn(recon, target)
                loss_list.append(_loss.item())
                loss += _loss
        return loss, loss_list

    def _train_one_epoch(self, train_loader, epoch):

        batch_loss, gen_epoch_loss, disc_epoch_loss, percept_epoch_loss, kl_epoch_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        losses_all_levels = []

        if self.verbose == 1:
            loop = tqdm(train_loader)
        elif self.verbose == 0:
            loop = train_loader

        self._model.train()
        self._discriminator.train()

        for step, data in enumerate(loop):
            _m = data.to(self.device)

            # Generator part
            self._optimizer.zero_grad()

            out = self.forward(_m)
            loss = out['loss'] 
            kl_loss = self.kl_divergence_loss(out['kl_loss'])
            recons = out['recon']
            p_loss = self.perceptual_loss(_m, recons)
            
            if self.kld_weight is not None:
                loss += self.kld_weight * kl_loss

            if self.perceptual_weight is not None:
                loss += self.perceptual_weight * p_loss

            if self.warmup_epochs is not None and epoch >= self.warmup_epochs:
                logits_fake = self._discriminator(recons[:,-1])[-1]
                generator_loss = self._adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss += self.adv_weight * generator_loss
        
            loss.backward()

            if self.gradient_clip and self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.gradient_clip_val)
            
            # Updates the parameters and adjust learning weights
            self._optimizer.step()

            if self.warmup_epochs is not None and epoch >= self.warmup_epochs:
                self._optimizer_d.zero_grad()
                logits_fake = self._discriminator(recons[:,-1].contiguous().detach())[-1]
                loss_d_fake = self._adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = self._discriminator(_m.contiguous().detach())[-1]
                loss_d_real = self._adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = self.adv_weight * discriminator_loss
                
                loss_d.backward()
                self._optimizer_d.step()

            self._scheduler.step()

            # Gather data and report
            batch_loss += loss.item()
            percept_epoch_loss += p_loss.item()
            kl_epoch_loss += kl_loss.item()

            if self.warmup_epochs is not None and epoch >= self.warmup_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            if len(losses_all_levels) == 0:
                losses_all_levels = [0.0]*len(out['loss_list'])
            
            for idx, loss_item in enumerate(out['loss_list']):
                losses_all_levels[idx] += loss_item
                    
            loss_list:list = out['loss_list']
            bar_info = f" Mixed: {loss_list[0]:.6f}"
            
            for idx, val in enumerate(loss_list[1:]):
                bar_info += f" | Level {idx+1}: {val:.6f}"

            bar_info += f" | Perceptual: {p_loss.item():.6f}"

            loss_dict = {
                "batch_loss": batch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
                "kl_loss": kl_epoch_loss / (step + 1), 
                "perceptual": percept_epoch_loss / (step + 1), 
                "info": bar_info
            }
            
            # Print loss log
            if self.verbose == 1:
                loop.set_description(f"Epoch [{epoch}/{self.num_epoch}]")
                loop.set_postfix(loss=loss.item(), info=bar_info)
            elif self.verbose == 0:
                pass

            loss_dict['all_levels_loss'] = [_/(step+1) for _ in losses_all_levels]
        return loss_dict

    def _save_model(self, path_to_ckpt):
        torch.save(
            {
                'autoencoder_state_dict': self._model.state_dict(),
                'discriminator_state_dict': self._discriminator.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
                'optimizer_d_state_dict': self._optimizer_d.state_dict()
            }, path_to_ckpt)

    def _load_model(self, path_to_ckpt, load_optimizer=False):
        if self.device == torch.device("cuda"):
            checkpoint = torch.load(join(path_to_ckpt, 'checkpoint.pt'))
        else:
            checkpoint = torch.load(join(path_to_ckpt, 'checkpoint.pt'), map_location=torch.device('cpu'))
        self._model.load_state_dict(checkpoint['autoencoder_state_dict'])
        self._discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if load_optimizer:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    def _print_loss(self, epoch, loss_dict, batch_vloss, all_vlosses):
        all_levels_loss = loss_dict['all_levels_loss']
        loss_info, vloss_info = f' Mixed: {all_levels_loss[0]:.6f}', f' Mixed: {all_vlosses[0]:.6f}'
        for idx, val in enumerate(all_levels_loss[1:]):
            loss_info += f" | Level {idx+1}: {val:.6f}"
        for idx, val in enumerate(all_vlosses[1:]):
            vloss_info += f" | Level {idx+1}: {val:.6f}"
        
        loss_info += ' | G loss: {:.6f}'.format(loss_dict['gen_loss'])
        loss_info += ' | D loss: {:.6f}'.format(loss_dict['disc_loss'])
        loss_info += ' | P loss: {:.6f}'.format(loss_dict['perceptual'])
        loss_info += ' | KL loss: {:.6f}'.format(loss_dict['kl_loss'])

        print("Epoch {:5s}: Train loss -- total {:.6f} |".format(str(epoch), loss_dict['batch_loss']) + loss_info)
        print("             Valid loss -- total {:.6f} |".format(batch_vloss) + vloss_info)
        print('         LR: {}'.format(self._scheduler.get_last_lr()))

    def train(self, 
              path_to_model, 
              train_loader, 
              valid_loader, 
              test_loader=None,
              if_track_validate=True, 
              ckpt_epoch:int=5,
              plot_test_recon:bool=False, 
              starting_epoch:int=0,
              ):

        train_loss_list, valid_loss_list, learning_rate_list = [], [], []
        epoch_gen_loss_list, epoch_disc_loss_list, epoch_perceptual_list = [], [], []

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(starting_epoch, self.num_epoch):
            
            # train
            _loss_dict = self._train_one_epoch(train_loader, epoch)
            train_loss_list.append(_loss_dict['batch_loss'])
            epoch_gen_loss_list.append(_loss_dict['gen_loss'])
            epoch_disc_loss_list.append(_loss_dict['disc_loss'])
            epoch_perceptual_list.append(_loss_dict['perceptual'])
            learning_rate_list.append(float(self._optimizer.param_groups[0]['lr']))

            # validate
            if if_track_validate == True:
                batch_vloss, all_vlosses = self._validate(valid_loader)
            else:
                batch_vloss, all_vlosses = 0.0, [0.0]
            valid_loss_list.append(batch_vloss)
            
            # test
            if plot_test_recon and test_loader is not None:
                test_true, test_recon, test_loss, loss_list_collect = self.test(test_loader)
                self.plot_recons(test_recon, test_true, join(path_to_model, f"test_recon_epoch_{epoch}.png"))
                self.plot_hists(loss_list_collect, join(path_to_model, f"test_distribution_epoch_{epoch}.png"))

            # print losses 
            self._print_loss(epoch, _loss_dict, batch_vloss, all_vlosses)

            # Save checkpoint every ckpt_epoch epochs
            if (epoch+1)%ckpt_epoch == 0:
                self._save_model(join(path_to_model, 'checkpoint{}.pt'.format(epoch)))
            
            # Save checkpoint every epoch
            self._save_model(join(path_to_model, 'checkpoint.pt'))

            # save losses
            np.savez(join(path_to_model, 'loss_epoch.npz'),
                     train_loss=np.array(train_loss_list),
                     valid_loss=np.array(valid_loss_list),
                     gen_loss_list=np.array(epoch_gen_loss_list), 
                     disc_loss_list=np.array(epoch_disc_loss_list), 
                     perceptual_list=np.array(epoch_perceptual_list),
                     learning_rate=np.array(learning_rate_list)
                     )

        return train_loss_list, valid_loss_list


