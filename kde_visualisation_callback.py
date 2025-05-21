import PIL.Image
if not hasattr(PIL.Image, "Resampling"):
    class _Resampling:             # minimal stub enum
        LANCZOS = PIL.Image.ANTIALIAS
    PIL.Image.Resampling = _Resampling
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from io import BytesIO
import PIL.Image
from pytorch_lightning.callbacks import Callback
from matplotlib.backends.backend_agg import FigureCanvasAgg
from utils import wasserstein_distance
from scipy.stats import wasserstein_distance as WD


class SampleVisualizationCallback(Callback):
    def __init__(self,config):
        super().__init__()
        self.config = config
        # self.causal_mask = causal_mask

    def on_test_end(self, trainer, pl_module):
        pl_module.eval()
        log_dir = trainer.logger.log_dir

        # ======= CausalNF Sampling =======
        # Get flow module from the model
        flow_module = pl_module.flow()  # Assuming pl_module is an instance of CausalNF

        with torch.no_grad():
            base_sample = flow_module.base.sample((self.config['train_samples'],))
            samples = flow_module.transform(base_sample).cpu().numpy()

        
        # # ======= Load a batch from validation loader =======
        # test_loader = trainer.datamodule.train_loader()
        # test_batch = next(iter(test_loader))

        # if isinstance(test_batch, (tuple, list)):
        #     test_samples = test_batch[0].cpu().numpy()
        # else:
        #     test_samples = test_batch.cpu().numpy()

        train_df = pd.read_csv(self.config['train_data'])
        train_df.drop(columns=self.config['target'],inplace=True)
        train_samples = train_df.values

        total_wd = wasserstein_distance(torch.tensor(samples),torch.tensor(train_samples))
        # Log total WD
        trainer.logger.experiment.add_scalar("metrics/total_wasserstein", total_wd, global_step=trainer.global_step)

        # ======= KDE Plot Per Dimension =======
        num_dims = train_samples.shape[1]
        fig, axes = plt.subplots(1, num_dims, figsize=(4 * num_dims, 4))

        for i in range(num_dims):
            wd = WD(
                torch.tensor(train_samples[:, i]), torch.tensor(samples[:, i])
                )
            sns.kdeplot(train_samples[:, i], label="Original", ax=axes[i], color="blue")
            sns.kdeplot(samples[:, i], label="Sampled", ax=axes[i], color="orange")
            axes[i].set_title(f"Dimension {i+1}\nWD = {wd:.4f}")
            axes[i].legend()

        plt.tight_layout()
        self._log_plot(trainer.logger.experiment, fig, "KDE_Plot", trainer.global_step)

        # ======= PCA + KDE + Scatter =======
        pca = PCA(n_components=2,svd_solver="full")
        pca.fit(samples)
        val_samples_2d = pca.transform(train_samples)
        samples_2d = pca.transform(samples)

        df_samples = pd.DataFrame(samples_2d, columns=['PCA1', 'PCA2'])
        df_samples['Type'] = 'Generated'

        df_val = pd.DataFrame(val_samples_2d, columns=['PCA1', 'PCA2'])
        df_val['Type'] = 'Original'

        df_all = pd.concat([df_val, df_samples], ignore_index=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        palette = {'Generated': 'red', 'Original': 'blue'}

        sns.kdeplot(
            data=df_all,
            x='PCA1', y='PCA2',
            hue='Type',
            fill=True,
            common_norm=False,
            alpha=0.4,
            palette=palette,
            levels=10,
            thresh=0.05,
            ax=ax
        )

        for label in ['Generated', 'Original']:
            subset = df_all[df_all['Type'] == label]
            sns.scatterplot(
                x=subset['PCA1'], y=subset['PCA2'],
                color=palette[label],
                label=label,
                s=10, alpha=0.5,
                ax=ax
            )

        ax.set_title("2D PCA Projection with KDE (Generated vs Original Samples)")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend(title='Data Type')
        plt.tight_layout()

        self._log_plot(trainer.logger.experiment, fig, "PCA_Projection", trainer.global_step)

    # def _log_plot(self, writer: SummaryWriter, fig, tag, global_step: int):
    #     buf = BytesIO()
    #     fig.savefig(buf, format='png')
    #     buf.seek(0)
    #     image = PIL.Image.open(buf)
    #     image_tensor = transforms.ToTensor()(image)
    #     writer.add_image(tag, image_tensor, global_step=global_step)
    #     plt.close(fig)    


    def _log_plot(self, writer, fig, tag, step):
        # Render the figure on a canvas without going through PIL
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape(h, w, 3)          # HWC
        writer.add_image(tag, img, global_step=step, dataformats='HWC')
        plt.close(fig)
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     # Use current model instead of re-loading from checkpoint
    #     model = pl_module.eval()
    #     log_dir = trainer.logger.log_dir

    #     with torch.no_grad():
    #         samples = model.sample(300).cpu().numpy()

    #     # Load a batch from validation loader
    #     val_loader = trainer.datamodule.val_dataloader()
    #     val_batch = next(iter(val_loader))

    #     # Assume val_batch is either (x, y) or just x
    #     if isinstance(val_batch, (tuple, list)):
    #         val_samples = val_batch[0].cpu().numpy()
    #     else:
    #         val_samples = val_batch.cpu().numpy()  


    #     num_dims = val_samples.shape[1]
    #     fig, axes = plt.subplots(1, num_dims, figsize=(4 * num_dims, 4))

    #     for i in range(num_dims):
    #         sns.kdeplot(val_samples[:, i], label="Validation", ax=axes[i], color="blue")
    #         sns.kdeplot(samples[:, i], label="Sampled", ax=axes[i], color="orange")
    #         axes[i].set_title(f"Dimension {i+1}")
    #         axes[i].legend()

    #     plt.tight_layout()
    #     self._log_plot(trainer.logger.experiment, fig, "KDE_Plot", trainer.global_step)

    #     # ===== PCA + KDE + Scatter =====
    #     pca = PCA(n_components=2,svd_solver='full')#,random_state=self.config['seed'])
    #     samples_2d = pca.fit_transform(val_samples)
    #     val_samples_2d = pca.transform(samples)

    #     # samples_2d = pca.fit_transform(samples)
    #     # val_samples_2d = pca.transform(val_samples)

    #     # samples_2d = pca.fit_transform(samples)
    #     # val_samples_2d = pca.fit_transform(val_samples)

    #     df_samples = pd.DataFrame(samples_2d, columns=['PCA1', 'PCA2'])
    #     df_samples['Type'] = 'Generated'

    #     df_val = pd.DataFrame(val_samples_2d, columns=['PCA1', 'PCA2'])
    #     df_val['Type'] = 'Validation'

    #     df_all = pd.concat([df_val,df_samples], ignore_index=True)

    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     palette = {'Generated': 'red', 'Validation': 'blue'}

    #     sns.kdeplot(
    #         data=df_all,
    #         x='PCA1', y='PCA2',
    #         hue='Type',
    #         fill=True,
    #         common_norm=False,
    #         alpha=0.4,
    #         palette=palette,
    #         levels=10,
    #         thresh=0.05,
    #         ax=ax
    #     )

    #     for label in ['Generated', 'Validation']:
    #         subset = df_all[df_all['Type'] == label]
    #         sns.scatterplot(
    #             x=subset['PCA1'], y=subset['PCA2'],
    #             color=palette[label],
    #             label=label,
    #             s=10, alpha=0.5,
    #             ax=ax
    #         )

    #     ax.set_title("2D PCA Projection with KDE (Generated vs Validation Samples)")
    #     ax.set_xlabel("PCA 1")
    #     ax.set_ylabel("PCA 2")
    #     ax.legend(title='Data Type')
    #     plt.tight_layout()

    #     self._log_plot(trainer.logger.experiment, fig, "PCA_Projection", trainer.global_step)