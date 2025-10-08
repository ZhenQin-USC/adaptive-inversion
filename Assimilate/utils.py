import yaml
import torch
import os
import numpy as np

from os.path import join

from .gradientbased_proxy import OptimizationStage, AdaptiveMask
from .latent_selector import MultiResolutionLatentSelector


def filter_non_decreasing(arr):
    if len(arr) == 0:
        return np.array([])
    filtered = [arr[0]] 
    for val in arr[1:]:
        if val < filtered[-1]: 
            filtered.append(val)
    return np.array(filtered)


def convert_stage_x(stage1, stage2, all_grain_factors, device):
    """
    Convert the optimized x from stage 1 into the corresponding x for stage 2.

    Args:
        stage1: The first stage optimization object.
        stage2: The second stage optimization object.
        all_grain_factors (list): List of downsampling factors [(dx, dy, dz), ...].
        device: Torch device (cuda or cpu).

    Returns:
        torch.Tensor: The converted x for stage 2.
    """

    # Extract x1 from the last iteration of stage 1
    x1m0 = stage1.opt_history['hx'][-1].reshape(1, -1)  # x1 with mask0
    x1m0_tensor = torch.tensor(x1m0, dtype=torch.float32, requires_grad=True).to(device)
    print(x1m0_tensor.shape)
    # Convert x to latent representation using stage1's optimizer
    z1m0 = stage1.proxy_optimizer._ensemble_to_latents(x1m0_tensor)
    print(f"Stage 1: x1m0_tensor.shape = {x1m0_tensor.shape}, z1m0.shape = {z1m0.shape}")

    # Apply different grain factors (multi-resolution downsampling)
    h = [z1m0] + [z1m0[:, :, ::dx, ::dy, ::dz] for (dx, dy, dz) in all_grain_factors]
    print(f"Multi-resolution shapes: {[_.shape for _ in h]}")

    # Select elements using stage 2's selector and mask
    x1m1_tensor = stage2.proxy_optimizer.selector.select_elements(h, stage2.proxy_optimizer.selected_mask)

    # Convert selected elements to latent space for stage 2
    z1m1 = stage2.proxy_optimizer._ensemble_to_latents(x1m1_tensor)
    print(f"Stage 2: x1m1_tensor.shape = {x1m1_tensor.shape}, z1m1.shape = {z1m1.shape}")

    # Compare latent space differences
    print(f"Latent space diff min: {(z1m0 - z1m1).min()}, max: {(z1m0 - z1m1).max()}")

    # Convert x1m1_tensor to numpy array
    x1 = x1m1_tensor.detach().cpu().numpy()
    return x1, z1m0, z1m1

    
def save_configs(save_dir, 
                 inverse_config, proxy_config, autoencoder_config, optimizer_config, 
                 *stage_configs
):
    """
    Saves configuration names and stage configurations into YAML files.
    
    Args:
        save_dir (str): Directory to save configuration files.
        inverse_config (str): Name of the inverse config.
        proxy_config (str): Name of the proxy config.
        autoencoder_config (str): Name of the autoencoder config.
        optimizer_config (str): Name of the optimizer config.
        *stage_configs (dict): Variable number of stage configurations.
    """

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save configuration names
    config_names = {
        "inverse_config": inverse_config,
        "proxy_config": proxy_config,
        "autoencoder_config": autoencoder_config,
        "optimizer_config": optimizer_config,
        "num_stages": len(stage_configs)  # Store the number of stages
    }
    config_names_path = join(save_dir, f"config_names.yaml")
    with open(config_names_path, "w") as f:
        yaml.dump(config_names, f)

    print(f"✅ Saved: {config_names_path}")

    # Save each stage configuration
    for i, stage_config in enumerate(stage_configs, start=1):
        stage_config_path = join(save_dir, f"config_stage_{i}.yaml")
        with open(stage_config_path, "w") as f:
            yaml.dump(stage_config, f)
        print(f"✅ Saved: {stage_config_path}")


def multi_level_single_stage_optimization(configs, 
                                          z0, 
                                          dir_to_results, 
                                          data_loader, 
                                          predictor, 
                                          autoencoder, 
                                          device, 
                                          opt_config, 
                                          update_map, 
                                          x0=None, 
                                          prev_optimizer=None,
                                          use_multi_level=True
                                         ):
    
    all_optimizers = {}
    curr_update_map = None

    for optimizer_name, optimizer_config in configs.items():
        # create latent selector
        latent_selector = MultiResolutionLatentSelector(
            optimizer_config['Basic setup']['all_grain_factors'], 
            optimizer_config['Basic setup']['all_latent_shapes']
            )   
        
        # get the update map for the current optimizer
        if isinstance(update_map, dict):
            if optimizer_name in update_map:
                curr_update_map = update_map.get(optimizer_name, None)
            else:
                if curr_update_map is None:
                    raise ValueError(f"❌ update_map for optimizer '{optimizer_name}' not found, and no previous update_map is available.")
                else:
                    print(f"⚠️ update_map for '{optimizer_name}' not found, using previous update_map.")
        else:
            curr_update_map = update_map # use the same update_map for all optimizers

        # create optimization object
        curr_optimizer = OptimizationStage(
            optimizer_config, data_loader, predictor, autoencoder, 
            latent_selector, device, opt_config, 
            masker=AdaptiveMask(z0, update_map=curr_update_map)
            )
        
        if x0 is None and prev_optimizer is None:    
            x0 = curr_optimizer.proxy_optimizer.calculate_zm(curr_optimizer.m_ens) # Compute initial latent variables
            x0 = 0.01 * np.ones_like(x0)
        elif prev_optimizer is not None:
            x0 = convert_stage_x(prev_optimizer, 
                                 curr_optimizer, 
                                 optimizer_config['Basic setup']['all_grain_factors'], 
                                 device)[0]
        else:
            _ = curr_optimizer.proxy_optimizer.calculate_zm(curr_optimizer.m_ens) # Initialize selector
        
        # Run optimization
        print(f"\n🔵 Running Optimization for Level {optimizer_name} ...")
        curr_optimizer.run_optimization(x0) 
        curr_optimizer.save_results(join(dir_to_results, optimizer_name)) # Save results

        # Memory cleanup after Stage 1
        curr_optimizer.sim_history['s'] = None
        curr_optimizer.sim_history['p'] = None
        torch.cuda.empty_cache()

        # Store outputs
        all_optimizers[optimizer_name] = curr_optimizer

        # Reset variables
        if use_multi_level: # pass the previous results to the next level
            z0 = torch.tensor(curr_optimizer.sim_history['z'][-1]).to(device)
            prev_optimizer = curr_optimizer
        else: # each level is independent of other levels
            x0 = None # Reset x0 to force using prev_optimizer next loop

    return all_optimizers