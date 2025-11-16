@dataclass
class VAEArgs():
    # Improved beta_kl - higher value prevents posterior collapse
    beta_kl: float = 0.01  # Changed from 1e-3 to 0.01

    # architecture - larger dimensions for better capacity
    latent_dim_size: int = 256  # Changed from 128 to 256
    hidden_dim_size: int = 1024  # Changed from 512 to 1024

    # data / training
    dataset: Literal["MNIST", "CELEB"] = "CELEB"
    batch_size: int = 128  # Reduced from 512 to 128 for more stable training
    epochs: int = 20  # Increased from 15 to 20
    lr: float = 2e-4  # Slightly increased from 1e-4
    betas: tuple[float, float] = (0.5, 0.999)

    # logging
    use_wandb: bool = True
    wandb_project: str | None = "day5-autoencoder"
    wandb_name: str | None = None
    log_every_n_batches: int = 100  # Changed name to be clearer (per batch, not per step)

class VAE(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size

        # Improved encoder with deeper architecture
        self.encoder = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),  # 64->32
            BatchNorm2d(32),
            ReLU(),
            Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # 32->16
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),  # 16->8
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),  # 8->4
            BatchNorm2d(256),
            ReLU(),
            Rearrange("b c h w -> b (c h w)"),
            Linear(256*4*4, hidden_dim_size),
            ReLU(),
            Linear(hidden_dim_size, 2*latent_dim_size),
            Rearrange("b (two l) -> two b l", two=2)
        )
        # Improved decoder with deeper architecture and Tanh output
        self.decoder = Sequential(
            Linear(latent_dim_size, hidden_dim_size),
            ReLU(),
            Linear(hidden_dim_size, 256*4*4),
            Rearrange("b (c h w) -> b c h w", c=256, h=4, w=4),
            ReLU(),
            ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),  # 4->8
            BatchNorm2d(128),
            ReLU(),
            ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),  # 8->16
            BatchNorm2d(64),
            ReLU(),
            ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),  # 16->32
            BatchNorm2d(32),
            ReLU(),
            ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),  # 32->64
            Tanh()  # Output in [-1, 1] to match data normalization
        )

    def sample_latent_vector(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Passes `x` through the encoder, returns tuple of (sampled latent vector, mean, log std dev).
        This function can be used in `forward`, but also used on its own to generate samples for
        evaluation.
        """
        (mu, logsigma) = self.encoder(x) # each has dim (1, batch_size, latent_dim_size)
        sigma = t.exp(logsigma)
        eps = t.randn_like(sigma)
        z = mu + sigma * eps
        return z, mu, logsigma

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Passes `x` through the encoder and decoder. Returns the reconstructed input, as well as mu
        and logsigma.
        """
        (z, mu, logsigma) = self.sample_latent_vector(x)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logsigma

class VAEVisualizer:
    """Utility class for VAE visualizations"""

    @staticmethod
    def tensor_to_image(tensor: Tensor) -> np.ndarray:
        """Convert a tensor in [-1, 1] to numpy array in [0, 1]"""
        img = tensor.detach().cpu()
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        # If batch, take first image
        if img.ndim == 4:
            img = img[0]
        # Convert from CxHxW to HxWxC
        if img.shape[0] in [1, 3]:
            img = img.permute(1, 2, 0)
        return img.numpy()

    @staticmethod
    def plot_reconstructions(original: Tensor, reconstructed: Tensor, n_images: int = 8,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot original images alongside their reconstructions"""
        fig, axes = plt.subplots(2, n_images, figsize=(2*n_images, 4))

        for i in range(n_images):
            # Original
            img_orig = VAEVisualizer.tensor_to_image(original[i])
            axes[0, i].imshow(img_orig)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            # Reconstructed
            img_recon = VAEVisualizer.tensor_to_image(reconstructed[i])
            axes[1, i].imshow(img_recon)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_samples(samples: Tensor, n_rows: int = 4, title: str = "Generated Samples",
                     save_path: Optional[str] = None) -> plt.Figure:
        """Plot a grid of generated samples"""
        n_samples = samples.shape[0]
        n_cols = (n_samples + n_rows - 1) // n_rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

        for i in range(n_samples):
            img = VAEVisualizer.tensor_to_image(samples[i])
            axes[i].imshow(img)
            axes[i].axis('off')

        # Hide extra subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_interpolation(images: Tensor, title: str = "Latent Space Interpolation",
                          save_path: Optional[str] = None) -> plt.Figure:
        """Plot interpolation between two images"""
        n_steps = images.shape[0]
        fig, axes = plt.subplots(1, n_steps, figsize=(2*n_steps, 2))

        for i in range(n_steps):
            img = VAEVisualizer.tensor_to_image(images[i])
            axes[i].imshow(img)
            axes[i].axis('off')

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_latent_space_2d(latent_vectors: Tensor, labels: Optional[Tensor] = None,
                            highlight_points: Optional[Tensor] = None,
                            title: str = "Latent Space (First 2 Dimensions)",
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot 2D projection of latent space"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Main scatter plot
        latent_np = latent_vectors[:, :2].cpu().numpy()
        if labels is not None:
            scatter = ax.scatter(latent_np[:, 0], latent_np[:, 1],
                               c=labels.cpu().numpy(), alpha=0.5, s=1, cmap='tab10')
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(latent_np[:, 0], latent_np[:, 1], alpha=0.3, s=1, c='blue')

        # Highlight specific points if provided
        if highlight_points is not None:
            highlight_np = highlight_points[:, :2].cpu().numpy()
            ax.scatter(highlight_np[:, 0], highlight_np[:, 1],
                      c='red', s=100, marker='x', linewidths=3,
                      label='Highlighted', zorder=5)
            ax.legend()

        ax.set_xlabel('Latent Dimension 0', fontsize=12)
        ax.set_ylabel('Latent Dimension 1', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_latent_grid(model: nn.Module, n_points: int = 11,
                        interpolation_range: tuple = (-3, 3), dims: tuple = (0, 1),
                        title: str = "Latent Space Grid",
                        save_path: Optional[str] = None) -> plt.Figure:
        """Generate and plot a grid traversal of latent space"""
        device = next(model.parameters()).device
        grid_latent = t.zeros(n_points, n_points, model.latent_dim_size, device=device)
        x = t.linspace(*interpolation_range, n_points)
        grid_latent[..., dims[0]] = x.unsqueeze(-1)
        grid_latent[..., dims[1]] = x
        grid_latent = grid_latent.flatten(0, 1)

        with t.inference_mode():
            output = model.decoder(grid_latent)

        # Reshape to grid
        output_grid = output.view(n_points, n_points, 3, 64, 64)
        output_grid = output_grid.permute(0, 3, 1, 4, 2)  # (rows, H, cols, W, C)
        output_grid = output_grid.reshape(n_points*64, n_points*64, 3)

        # Denormalize
        output_grid = (output_grid + 1) / 2
        output_grid = output_grid.clamp(0, 1)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(output_grid.cpu().detach().numpy())
        ax.axis('off')
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

class VAETrainer:
    def __init__(self, args: VAEArgs):
        self.args = args
        self.trainset = get_dataset(args.dataset)
        self.trainloader = DataLoader(
            self.trainset, batch_size=args.batch_size, shuffle=True, num_workers=8
        )
        self.model = VAE(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)
        # Add learning rate scheduler - reduce LR by 0.5 every 5 epochs
        self.scheduler = t.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.visualizer = VAEVisualizer()

        # Track losses for plotting
        self.loss_history = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'batch_idx': []
        }

    def training_step(self, img: Tensor):
        """
        Performs a training step on the batch of images in `img`. Returns the loss. Logs to wandb
        if enabled.
        """
        img = img.to(device)
        recon_img, mu, logsigma = self.model(img)

        # Use L1 loss which works better for images than MSE
        recon_loss = F.l1_loss(recon_img, img)

        # Correct KL divergence formula: KL[q(z|x) || N(0,1)] = 0.5 * sum(1 + log(σ²) - μ² - σ²)
        # Since logsigma = log(σ), we have log(σ²) = 2*log(σ) = 2*logsigma
        kl_div = 0.5 * (t.exp(2*logsigma) + mu**2 - 1 - 2*logsigma)
        kl_loss = kl_div.sum(dim=-1).mean()
        loss = recon_loss + self.args.beta_kl * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += img.shape[0]

        # Track losses
        self.loss_history['total_loss'].append(loss.item())
        self.loss_history['recon_loss'].append(recon_loss.item())
        self.loss_history['kl_loss'].append(kl_loss.item())
        self.loss_history['batch_idx'].append(self.batch_idx)

        if self.args.use_wandb:
            wandb.log(
                dict(
                    reconstruction_loss=recon_loss.item(),
                    kl_div_loss=kl_loss.item(),
                    mean=mu.mean().item(),
                    std=t.exp(logsigma).mean().item(),
                    total_loss=loss.item(),
                ),
                step=self.step,
            )
        return loss

    @t.inference_mode()
    def log_samples(self) -> None:
        """
        Evaluates model on holdout data, either logging to wandb or displaying output inline.
        """
        assert self.step > 0, (
            "First call should come after a training step. Remember to increment `self.step`."
        )
        output = self.model(HOLDOUT_DATA_CELEB)[0]

        if self.args.use_wandb:
            # Log reconstruction comparison
            fig_recon = self.visualizer.plot_reconstructions(
                HOLDOUT_DATA_CELEB, output, n_images=8
            )
            wandb.log({"reconstructions": wandb.Image(fig_recon)}, step=self.step)
            plt.close(fig_recon)

            # Generate and log random samples
            z = t.randn(16, self.model.latent_dim_size, device=device)
            samples = self.model.decoder(z)
            fig_samples = self.visualizer.plot_samples(samples, n_rows=4, title="Random Samples")
            wandb.log({"random_samples": wandb.Image(fig_samples)}, step=self.step)
            plt.close(fig_samples)
        else:
            # Display reconstructions
            fig_recon = self.visualizer.plot_reconstructions(
                HOLDOUT_DATA_CELEB, output, n_images=8
            )
            plt.show()

            # Display random samples
            z = t.randn(16, self.model.latent_dim_size, device=device)
            samples = self.model.decoder(z)
            fig_samples = self.visualizer.plot_samples(samples, n_rows=4, title="Random Samples")
            plt.show()

    def plot_loss_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot training loss curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Total loss
        axes[0].plot(self.loss_history['batch_idx'], self.loss_history['total_loss'],
                    alpha=0.6, linewidth=0.5)
        # Add moving average
        window = 50
        if len(self.loss_history['total_loss']) > window:
            moving_avg = np.convolve(self.loss_history['total_loss'],
                                    np.ones(window)/window, mode='valid')
            axes[0].plot(self.loss_history['batch_idx'][window-1:], moving_avg,
                        'r-', linewidth=2, label=f'{window}-batch moving avg')
            axes[0].legend()
        axes[0].set_xlabel('Batch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].grid(True, alpha=0.3)

        # Reconstruction loss
        axes[1].plot(self.loss_history['batch_idx'], self.loss_history['recon_loss'],
                    alpha=0.6, linewidth=0.5)
        if len(self.loss_history['recon_loss']) > window:
            moving_avg = np.convolve(self.loss_history['recon_loss'],
                                    np.ones(window)/window, mode='valid')
            axes[1].plot(self.loss_history['batch_idx'][window-1:], moving_avg,
                        'r-', linewidth=2, label=f'{window}-batch moving avg')
            axes[1].legend()
        axes[1].set_xlabel('Batch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Reconstruction Loss (L1)')
        axes[1].grid(True, alpha=0.3)

        # KL divergence loss
        axes[2].plot(self.loss_history['batch_idx'], self.loss_history['kl_loss'],
                    alpha=0.6, linewidth=0.5)
        if len(self.loss_history['kl_loss']) > window:
            moving_avg = np.convolve(self.loss_history['kl_loss'],
                                    np.ones(window)/window, mode='valid')
            axes[2].plot(self.loss_history['batch_idx'][window-1:], moving_avg,
                        'r-', linewidth=2, label=f'{window}-batch moving avg')
            axes[2].legend()
        axes[2].set_xlabel('Batch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('KL Divergence Loss')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @t.inference_mode()
    def comprehensive_evaluation(self, n_samples: int = 5000, save_dir: str = "vae_results"):
        """
        Perform comprehensive evaluation and generate all visualizations.
        This is called at the end of training.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print("Performing comprehensive evaluation...")
        print(f"{'='*60}\n")

        # 1. Plot loss curves
        print("1. Plotting loss curves...")
        fig_loss = self.plot_loss_curves(save_path=f"{save_dir}/loss_curves.png")
        if not self.args.use_wandb:
            plt.show()
        else:
            wandb.log({"final_loss_curves": wandb.Image(fig_loss)})
        plt.close(fig_loss)

        # 2. Reconstruction quality
        print("2. Evaluating reconstruction quality...")
        output = self.model(HOLDOUT_DATA_CELEB)[0]
        fig_recon = self.visualizer.plot_reconstructions(
            HOLDOUT_DATA_CELEB, output, n_images=8,
            save_path=f"{save_dir}/reconstructions.png"
        )
        if not self.args.use_wandb:
            plt.show()
        else:
            wandb.log({"final_reconstructions": wandb.Image(fig_recon)})
        plt.close(fig_recon)

        # 3. Random samples from prior
        print("3. Generating random samples from prior...")
        z = t.randn(16, self.model.latent_dim_size, device=device)
        samples = self.model.decoder(z)
        fig_samples = self.visualizer.plot_samples(
            samples, n_rows=4, title="Random Samples from Prior N(0,1)",
            save_path=f"{save_dir}/random_samples.png"
        )
        if not self.args.use_wandb:
            plt.show()
        else:
            wandb.log({"final_random_samples": wandb.Image(fig_samples)})
        plt.close(fig_samples)

        # 4. Latent space visualization
        print("4. Visualizing latent space...")
        small_dataset = Subset(self.trainset, indices=range(0, min(n_samples, len(self.trainset))))
        imgs = t.stack([img for img, label in small_dataset]).to(device)
        latent_vectors = self.model.encoder(imgs)[0]  # Get mu
        holdout_latent = self.model.encoder(HOLDOUT_DATA_CELEB)[0]

        fig_latent = self.visualizer.plot_latent_space_2d(
            latent_vectors, highlight_points=holdout_latent,
            title="Latent Space Distribution (First 2 Dimensions)",
            save_path=f"{save_dir}/latent_space_2d.png"
        )
        if not self.args.use_wandb:
            plt.show()
        else:
            wandb.log({"final_latent_space": wandb.Image(fig_latent)})
        plt.close(fig_latent)

        # 5. Latent space grid traversal
        print("5. Generating latent space grid traversal...")
        fig_grid = self.visualizer.plot_latent_grid(
            self.model, n_points=11, interpolation_range=(-3, 3),
            title="Latent Space Grid (Dimensions 0 and 1)",
            save_path=f"{save_dir}/latent_grid.png"
        )
        if not self.args.use_wandb:
            plt.show()
        else:
            wandb.log({"final_latent_grid": wandb.Image(fig_grid)})
        plt.close(fig_grid)

        # 6. Interpolations between images
        print("6. Generating interpolations...")
        n_interp = 10
        img1, img2 = HOLDOUT_DATA_CELEB[0], HOLDOUT_DATA_CELEB[1]
        z1 = self.model.encoder(img1.unsqueeze(0))[0]
        z2 = self.model.encoder(img2.unsqueeze(0))[0]
        alphas = t.linspace(0, 1, n_interp, device=device)
        z_interp = t.stack([alpha * z2 + (1 - alpha) * z1 for alpha in alphas])
        interp_samples = self.model.decoder(z_interp.squeeze(1))

        fig_interp = self.visualizer.plot_interpolation(
            interp_samples, title="Latent Space Interpolation",
            save_path=f"{save_dir}/interpolation.png"
        )
        if not self.args.use_wandb:
            plt.show()
        else:
            wandb.log({"final_interpolation": wandb.Image(fig_interp)})
        plt.close(fig_interp)

        print(f"\n{'='*60}")
        print(f"Evaluation complete! Results saved to '{save_dir}/'")
        print(f"{'='*60}\n")

    def train(self):
        """Performs a full training run."""
        self.step = 0  # Total samples seen
        self.batch_idx = 0  # Total batches seen
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, mode="offline")
            wandb.watch(self.model)

        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            for (img, _) in tqdm(self.trainloader, desc=f"Epoch {epoch+1}"):
                loss = self.training_step(img)
                self.batch_idx += 1

                # Log samples at regular intervals (per batch, not per sample)
                if self.batch_idx % self.args.log_every_n_batches == 0:
                    self.log_samples()

            # Step the learning rate scheduler after each epoch
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} complete. Learning rate: {current_lr:.6f}")

        # Perform comprehensive evaluation after training
        self.comprehensive_evaluation(save_dir="vae_results")

        if self.args.use_wandb:
            wandb.finish()

        self.save_model('vae_celeba_final.pt')

        return self.model

    def save_model(self, path='vae_model.pt'):
        """Save the trained model"""
        t.save({
            'model_state_dict': self.model.state_dict(),
            'args': self.args,
        }, path)
        print(f"Model saved to {path}")


def evaluate_saved_model(model_path: str, latent_dim_size: int, hidden_dim_size: int,
                        save_dir: str = "vae_evaluation"):
    """
    Load a saved VAE model and perform comprehensive evaluation with visualizations.

    Args:
        model_path: Path to the saved model checkpoint
        latent_dim_size: Latent dimension size used during training
        hidden_dim_size: Hidden dimension size used during training
        save_dir: Directory to save evaluation results
    """
    print(f"Loading model from {model_path}...")
    vae = VAE(latent_dim_size=latent_dim_size, hidden_dim_size=hidden_dim_size).to(device)

    # Register VAEArgs as safe for unpickling (PyTorch 2.6+ requirement)
    import torch.serialization
    torch.serialization.add_safe_globals([VAEArgs])

    checkpoint = t.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['model_state_dict'])
    else:
        vae.load_state_dict(checkpoint)

    vae.eval()
    print("Model loaded successfully!")

    # Create a visualizer
    visualizer = VAEVisualizer()

    import os
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Performing evaluation...")
    print(f"{'='*60}\n")

    # 1. Reconstruction quality
    print("1. Evaluating reconstruction quality...")
    with t.inference_mode():
        output = vae(HOLDOUT_DATA_CELEB)[0]
    fig_recon = visualizer.plot_reconstructions(
        HOLDOUT_DATA_CELEB, output, n_images=8,
        save_path=f"{save_dir}/reconstructions.png"
    )
    plt.show()
    plt.close(fig_recon)

    # 2. Random samples from prior
    print("2. Generating random samples from prior...")
    with t.inference_mode():
        z = t.randn(16, vae.latent_dim_size, device=device)
        samples = vae.decoder(z)
    fig_samples = visualizer.plot_samples(
        samples, n_rows=4, title="Random Samples from Prior N(0,1)",
        save_path=f"{save_dir}/random_samples.png"
    )
    plt.show()
    plt.close(fig_samples)

    # 3. Latent space visualization
    print("3. Visualizing latent space...")
    trainset = get_dataset("CELEB")
    small_dataset = Subset(trainset, indices=range(0, min(5000, len(trainset))))
    imgs = t.stack([img for img, label in small_dataset]).to(device)
    with t.inference_mode():
        latent_vectors = vae.encoder(imgs)[0]  # Get mu
        holdout_latent = vae.encoder(HOLDOUT_DATA_CELEB)[0]

    fig_latent = visualizer.plot_latent_space_2d(
        latent_vectors, highlight_points=holdout_latent,
        title="Latent Space Distribution (First 2 Dimensions)",
        save_path=f"{save_dir}/latent_space_2d.png"
    )
    plt.show()
    plt.close(fig_latent)

    # 4. Latent space grid traversal
    print("4. Generating latent space grid traversal...")
    fig_grid = visualizer.plot_latent_grid(
        vae, n_points=11, interpolation_range=(-3, 3),
        title="Latent Space Grid (Dimensions 0 and 1)",
        save_path=f"{save_dir}/latent_grid.png"
    )
    plt.show()
    plt.close(fig_grid)

    # 5. Interpolations between images
    print("5. Generating interpolations...")
    n_interp = 10
    img1, img2 = HOLDOUT_DATA_CELEB[0], HOLDOUT_DATA_CELEB[1]
    with t.inference_mode():
        z1 = vae.encoder(img1.unsqueeze(0))[0]
        z2 = vae.encoder(img2.unsqueeze(0))[0]
    alphas = t.linspace(0, 1, n_interp, device=device)
    z_interp = t.stack([alpha * z2 + (1 - alpha) * z1 for alpha in alphas])
    with t.inference_mode():
        interp_samples = vae.decoder(z_interp.squeeze(1))

    fig_interp = visualizer.plot_interpolation(
        interp_samples, title="Latent Space Interpolation",
        save_path=f"{save_dir}/interpolation.png"
    )
    plt.show()
    plt.close(fig_interp)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to '{save_dir}/'")
    print(f"{'='*60}\n")

    return vae
