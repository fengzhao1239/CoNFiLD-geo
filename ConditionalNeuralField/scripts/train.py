from os.path import exists
from os import mkdir
import os
import sys
from basicutility import ReadInput as ri
import neptune
import matplotlib.pyplot as plt


from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from einops import rearrange

from cnf.utils.normalize import Normalizer_ts
from cnf.utils import readdata
from cnf import nf_networks
from functools import partial
from cnf import visualize_tools

class basic_set(Dataset):
    def __init__(self, fois, coord):        # extra_siren_in is Reynolds number, which will not be used
        super().__init__()
        self.fois = fois    # field of interests
        self.total_samples = fois.shape[0]
        
        self.coords = coord

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return self.coords, self.fois[idx], idx


class LatentContainer(torch.nn.Module):
    """
    a model container that stores latents for multi GPU
    like a dataset align with the training samples (N_samples = N_training_samples)
    """

    def __init__(
        self,
        N_samples,
        N_features,
        dims,
        init_mean=None,
        init_std=None,
        lumped=False
    ):
        super().__init__()
        self.expand_dims = " ".join(["1" for _ in range(dims)]) if not lumped else "1"
        self.expand_dims = f"N f -> N {self.expand_dims} f"
        if init_mean is not None and init_std is not None:
            init_mean = init_mean.view(1, -1).cpu()
            init_std = init_std.view(1, -1).cpu()
            self.latents = torch.nn.Parameter(
                torch.randn((N_samples, N_features), dtype=torch.float32) * init_std + init_mean
            )
        else:
            self.latents = torch.nn.Parameter(
                torch.zeros((N_samples, N_features), dtype=torch.float32)
            )

    def forward(self, batch_ids):
        return rearrange(self.latents[batch_ids], self.expand_dims)

def compute_kl_loss_full(latent, eps=1e-8):
    mean = latent.mean(dim=0)  # (latent_dim,)
    std = latent.std(dim=0, unbiased=False) + eps  # (latent_dim,)
    kl_loss = 0.5 * (std**2 + mean**2 - 1 - torch.log(std**2 + eps)).mean()
    return kl_loss

def RMSE(prediction, target, dims=(1, 2)):

    return torch.sqrt(torch.mean(torch.abs(prediction - target) ** 2, dim=dims))

    # return torch.abs(prediction - target).mean(dim=dims) / torch.abs(target).mean(
    #     dim=dims
    # )


class trainer:

    def __init__(self, hyper_para: ri.basic_input, tag: str, infer_dps=False):
        '''
        Initialize the training module for the Conditional Neural Field model.
        For propogating gradient through the model (e.g. in DPS), set self.nf.eval()
        Args:
            hyper_para (basic_input): The hyperparameters for the model.
        '''
        self.hyper_para = hyper_para
        self.logger_tag = tag
        # Initialize basic attributes
        self._init_paths(hyper_para)
        
        # Load and process data (if not in inference mode)
        self._load_data(hyper_para)
        
        # Setup normalizers
        self._setup_normalizers(hyper_para)
        
        # Initialize network and related components
        self._init_network(hyper_para)
        
        # Configure inference mode if neede
        
        # Setup visualization and additional components
        # self._setup_visualization(hyper_para)
        # self._load_obstacle_mask(hyper_para)

    def _init_paths(self, hyper_para):
        '''Initialize basic attributes for the trainer.'''
        self.world_size = hyper_para.multiGPU
        if hasattr(hyper_para, "loading_path"):
            self.loading_path = hyper_para.loading_path
        else:
            self.loading_path = hyper_para.save_path
        
        if hasattr(hyper_para, "ckpt_loading_path"):
            self.ckpt_loading_path = hyper_para.ckpt_loading_path
        else:
            self.ckpt_loading_path = hyper_para.save_path
            
        if not exists(f"{hyper_para.save_path}"):
            os.makedirs(hyper_para.save_path, exist_ok=True)


# ---------------------------- Raw Data Curation ------------------------------------------

    def _load_data(self, hyper_para):
        '''Load and preprocess training data.'''
        # Load field data (fois)
        fois = self._load_field_data(hyper_para)
        self._validate_data_shape(fois, hyper_para)
        
        # Reshape the data according to batch shape
        # for dynamic "N t h w c" -> "(N t) h w c", flatten operation, get the tracjectory 'image' by sequential indexing
        # for static "N h w c" -> "N h w c", unchanged
        fois = rearrange(fois, f"{hyper_para.readin_data_shape} -> {hyper_para.batch_shape}")
        self._validate_output_features(fois, hyper_para)
        
        # Load coordinate data
        coord = self._load_coordinate_data(hyper_para)

        # Convert data to tensors if needed
        fois, coord = self._convert_to_tensors(fois, coord)
        
        # Store number of samples
        self.N_samples = fois.shape[0]
        print(f"total training samples: {self.N_samples}")
        
        # Store data for dataset creation later
        self.fois = fois           # <N, h, w, c>
        self.coord = coord         # <1, h, w, c=dims>

    def _load_field_data(self, hyper_para):
        '''Load the field data based on the configuration.'''
        if hasattr(hyper_para, "load_data_fn"):
            if type(hyper_para.load_data_fn) == str:
                load_data_fn = getattr(readdata, hyper_para.load_data_fn)
                load_params = {}
            elif type(hyper_para.load_data_fn) == dict:
                load_data_fn = getattr(readdata, hyper_para.load_data_fn["name"])
                load_params = hyper_para.load_data_fn["kwargs"]
            fois = load_data_fn(**load_params)
        else:
            raise NotImplementedError("load_data_fn must be provided.")
        return fois
    
    def _load_coordinate_data(self, hyper_para):
        '''Load coordinate data based on configuration.'''
        if hasattr(hyper_para, "load_coord_fn"):
            if type(hyper_para.load_coord_fn) == str:
                load_coord_fn = getattr(readdata, hyper_para.load_coord_fn)
                load_params = {}
            elif type(hyper_para.load_coord_fn) == dict:
                load_coord_fn = getattr(readdata, hyper_para.load_coord_fn["name"])
                load_params = hyper_para.load_coord_fn["kwargs"]
            coord = load_coord_fn(**load_params)
        else:
            raise NotImplementedError("load_coord_fn must be provided.")
        return coord

    def _validate_data_shape(self, fois, hyper_para):
        '''Validate that the data shape matches the expected shape.'''
        assert (rearrange(fois, f"{hyper_para.readin_data_shape} -> {hyper_para.readin_data_shape}") == fois).all(), \
            f"data shape is {tuple(fois.shape)}, which is inconsistant with the fois_shape ({hyper_para.readin_data_shape}) specified in yaml file."

    def _validate_output_features(self, fois, hyper_para):
        '''Validate that the output features match the neural field configuration.'''
        if "kwargs" in hyper_para.NF:
            assert (hyper_para.NF["kwargs"]["out_features"] == fois.shape[-1]), \
                f"NF_out_features is not consistent with fois shape, fion shape is {fois.shape}"
        else:
            assert (hyper_para.NF["out_features"] == fois.shape[-1]), \
                f"NF_out_features is not consistent with fois shape, fion shape is {fois.shape}"

    def _convert_to_tensors(self, fois, coord):
        '''Convert numpy arrays to PyTorch tensors if needed.'''
        fois = torch.tensor(fois, dtype=torch.float32) if not isinstance(fois, torch.Tensor) else fois
        
        if type(coord) == tuple or type(coord) == list:
            print("coord is tuple or list")
            coord = tuple(
                torch.tensor(i, dtype=torch.float32) if not isinstance(i, torch.Tensor) else i
                for i in coord
            )
        else:
            coord = torch.tensor(coord, dtype=torch.float32) if not isinstance(coord, torch.Tensor) else coord
        return fois, coord


# ---------------------------- Data Normalization ------------------------------------------

    def _setup_normalizers(self, hyper_para):
        '''Setup data normalizers for inputs and outputs.'''
        # Create normalizers
        self.in_normalizer = Normalizer_ts(**hyper_para.normalizer)    # class instance for input
        self.out_normalizer = Normalizer_ts(**hyper_para.normalizer)   # class instance for output
        
        # Load or fit normalizer parameters
        self._load_or_fit_normalizers(hyper_para)
        
        # Normalize data if not in inference mode
        if hasattr(self, "coord") and hasattr(self, "fois"):
            self._normalize_data()

    def _load_or_fit_normalizers(self, hyper_para):
        '''Load normalizer parameters if available, otherwise fit them.'''
        if exists(f"{self.loading_path}/normalizer_params.pt"):        # self.loading_path = save_path
            print(f"loading normalizer parameters from {self.loading_path}/normalizer_params.pt")
            norm_params = torch.load(f"{self.loading_path}/normalizer_params.pt")
            self.in_normalizer.params = norm_params["x_normalizer_params"]
            self.out_normalizer.params = norm_params["y_normalizer_params"]
        else:
            self._fit_normalizers(hyper_para)

    def _fit_normalizers(self, hyper_para):
        '''Fit normalizers to the data and save parameters.'''
        print("No noramlization file found! Calculating normalizer parameters and save.")

        # fit the flatten version, only get the updated params. e.g., min max values
        self.in_normalizer.fit_normalize(self.coord.flatten(0, hyper_para.dims-1))  # coord is <h,w,c> --> <(h,w),c> / <m,c> --> <m,c>
        self.out_normalizer.fit_normalize(self.fois.flatten(0, hyper_para.dims))    # fois is <N,h,w,c> --> <(N,h,w),c> / <N,m,c> --> <(N,m),c>
        print(f"Saving normalizer parameters to {hyper_para.save_path}/normalizer_params.pt")
        toSave = {
            "x_normalizer_params": self.in_normalizer.get_params(),
            "y_normalizer_params": self.out_normalizer.get_params(),
        }
        torch.save(toSave, hyper_para.save_path + "/normalizer_params.pt")

    def _normalize_data(self):
        '''Normalize the data using the fitted normalizers.'''
        self.normed_coords = self.in_normalizer.normalize(self.coord)
        self.normed_fois = self.out_normalizer.normalize(self.fois)
        assert self.normed_fois.shape == self.fois.shape, \
            f"normalized fois shape is {self.normed_fois.shape}, which is not consistent with the original fois shape {self.fois.shape}"
        assert self.normed_coords.shape == self.coord.shape, \
            f"normalized coords shape is {self.normed_coords.shape}, which is not consistent with the original coords shape {self.coord.shape}"


# ---------------------------- Neural Network & Dataset Initialization ------------------------------------------

    def _init_network(self, hyper_para):
        '''Initialize the neural field network and related components.'''
        # Initialize neural field
        self._init_neural_field(hyper_para)
        # self.load(-1, siren_only=True)  # ! ********** load the neural field network **********
        
        # Initialize latents if not in inference mode
        if hasattr(self, "latent_mean") and hasattr(self, "latent_std"):
            print(f"Initializing latents with transferred mean and std")
            self.latents = LatentContainer(self.N_samples, 
                                            hyper_para.hidden_size, 
                                            hyper_para.dims, 
                                            init_mean=self.latent_mean, 
                                            init_std=self.latent_std)       # the learnable latents
        else:
            print("Initializing latents with zeros")
            self.latents = LatentContainer(self.N_samples, 
                                            hyper_para.hidden_size, 
                                            hyper_para.dims)

        # Initialize dataset if not in inference mode
        self._init_dataset(hyper_para)

    def _init_neural_field(self, hyper_para):
        '''Initialize the neural field network.'''
        if "kwargs" not in hyper_para.NF:
            raise NotImplementedError("kwargs must be provided in config file!!!")
        else:
            self.nf = getattr(nf_networks, hyper_para.NF["name"])(**hyper_para.NF["kwargs"])      # the neural network (net1 + net2)
            total = sum(p.numel() for p in self.nf.parameters() if p.requires_grad)
            print(f"*** Total trainable parameters of neural field: {total}")

    def _init_dataset(self, hyper_para):
        '''Initialize the dataset for training.'''
        if hasattr(hyper_para, "dataset"):
            raise NotImplementedError
        else:
            self.dataset = basic_set(self.normed_fois, self.normed_coords)                       # the dataset
        
        self.test_criteria = partial(
            getattr(sys.modules[__name__], hyper_para.test_criteria),
        )


# ---------------------------- Training Loop ------------------------------------------

    def infer(
        self,
        coord: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        if coord is None:
            print("Using default training query points")
        coord = coord if coord is not None else self.train_coord
        coord = self.in_normalizer.normalize(coord)
        # print(f"in latent shape: {latents.shape}")
        if len(coord.shape) > 2:
            latents = latents[:, None, None]
        else:
            latents = latents[:, None]
        # print(f"Inference coord shape: {coord.shape}, latents shape: {latents.shape}")
        out = self.nf(coord.to(latents.device), latents)
        return self.out_normalizer.denormalize(out)


    def train(self, fix_nf=False):
        '''if fix_nf is True, the nf network will not be updated'''

        self.epoches = self.hyper_para.epochs
        self.criterion = getattr(torch.nn, self.hyper_para.loss_fn)()
        self.lr = self.hyper_para.lr
        self.save_dict = {
            "save_path": self.hyper_para.save_path,
            "save_every": self.hyper_para.save_every,
        }
        optim_dict = self.optim_dict if hasattr(self, "optim_dict") else {}
        start_epoch = self.start_epoch if hasattr(self, "start_epoch") else 0

        if self.world_size > 1:
            print(f"Data Parallel training, using {self.world_size} GPUs")

            p = mp.spawn(
                self._single_trainer,
                args=(
                    self.nf,
                    self.latents,
                    self.criterion,
                    self.lr,
                    self.dataset,
                    self.hyper_para,
                    self.save,
                    self.world_size,
                    optim_dict,
                    start_epoch,
                    self.out_normalizer,
                    self.test_criteria,
                    fix_nf,
                ),
                nprocs=self.world_size,
                join=False,
            )
            p.join()
        else:
            print('Single GPU training')
            self._single_trainer(
                0,
                self.nf,
                self.latents,
                self.criterion,
                self.lr,
                self.dataset,
                self.hyper_para,
                self.save,
                self.world_size,
                optim_dict,
                start_epoch,
                self.out_normalizer,
                self.test_criteria,
                fix_nf
            )

    @staticmethod
    def _single_trainer(
        rank,
        model,
        latents,
        criterion,
        learning_rate,
        dataset,
        hyper_para,
        savefn,
        world_size=1,
        optim_dict={},
        start_epoch=0,
        out_normalizer=None,
        test_criteria=None,
        fix_nf = False,
    ):

        model.to(rank)
        latents.to(rank)
        
        if world_size > 1:

            dist.init_process_group("nccl", rank=rank, world_size=world_size)

            model = DDP(model, device_ids=[rank])
            latents = DDP(latents, device_ids=[rank])

            train_loader = DataLoader(
                dataset,
                batch_size=hyper_para.batch_size,
                shuffle=False,
                sampler=DistributedSampler(dataset),
            )
            # test_loader = DataLoader(
            #     dataset,
            #     batch_size=hyper_para.test_batch_size,
            #     shuffle=False,
            #     sampler=DistributedSampler(dataset),
            # )
        else:
            train_loader = DataLoader(
                dataset, batch_size=hyper_para.batch_size, shuffle=True
            )
            # test_loader = DataLoader(
            #     dataset, batch_size=hyper_para.test_batch_size, shuffle=False
            # )

        if rank == 0:
            # logger = SummaryWriter(hyper_para.save_path)

            run = neptune.init_run(
                    project=hyper_para.neptune_project,
                    api_token=hyper_para.neptune_api_token,
                    tags=[hyper_para.log_tag],
                )
            run['config'] = hyper_para.yaml_dict

        # also load the optimizer state dict helps testing unseen samples (i.e., by transfer learning)
        optim_net_dec = torch.optim.AdamW(model.parameters(), lr=learning_rate["nf"])
        optim_states = torch.optim.AdamW(latents.parameters(), lr=learning_rate["latents"])
        
        net_dec_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optim_net_dec,
            mode="min",
            factor=0.5,
            patience=300,
            cooldown=0,
            threshold=5e-4,
            min_lr=1e-8,
        )
        
        states_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optim_states,
            mode="min",
            factor=0.5,
            patience=300,
            cooldown=0,
            threshold=5e-4,
            min_lr=1e-8,
        )

        for k in optim_dict:
            if k == "optim_net_dec_dict":
                optim_net_dec.load_state_dict(optim_dict["optim_net_dec_dict"])
            elif k == "optim_states_dict":
                optim_states.load_state_dict(optim_dict["optim_states_dict"])
        
        if fix_nf:
            # not update nf params
            model.eval()
        
        for i in tqdm(range(start_epoch, start_epoch + hyper_para.epochs)):
            if world_size > 1:
                train_loader.sampler.set_epoch(i)
            if not fix_nf:
                optim_net_dec.step()
                optim_net_dec.zero_grad()                 # update the nf network only once every epoch
            
            train_ins_error = []

            for batch_coords, batch_fois, idx in train_loader:
                batch_latent = latents(idx)               # get the latent
                batch_fois = batch_fois.to(rank)
                batch_coords = batch_coords.to(rank)
                
                batch_output = model(batch_coords, batch_latent)
                kl_loss = compute_kl_loss_full(batch_latent)    # ! kl loss
                loss = criterion(batch_output, batch_fois)
                optim_states.zero_grad()
                loss.backward()
                optim_states.step()                       # update the latent every mini batch

                train_ins_error.append(loss)              # mini batch loss

                if rank == 0:
                    run["train/mini_batch_loss"].append(loss.item())
                    run["train/kl_loss"].append(kl_loss.item())

            epoch_loss = torch.stack(train_ins_error).mean()  # epoch loss
            if world_size > 1:
                torch.distributed.reduce(
                    epoch_loss, op=torch.distributed.ReduceOp.AVG, dst=0
                )
                
            net_dec_scheduler.step(epoch_loss)
            states_scheduler.step(epoch_loss)
            
            if rank == 0:
                mean_loss = epoch_loss.item()
                # tqdm.write(str(mean_loss))
                # logger.add_scalar("loss", mean_loss, i)
                run["train/loss"].append(mean_loss)
                run["train/nf_lr"].append(optim_net_dec.param_groups[0]['lr'])
                run["train/latent_lr"].append(optim_states.param_groups[0]['lr'])
            
            if i % hyper_para.plot_every == 0 and i > 0:
                if rank == 0:
                    inference = out_normalizer.denormalize(batch_output)[0]
                    target = out_normalizer.denormalize(batch_fois)[0]
                    channel = inference.shape[-1]
                    fig, axs = plt.subplots(channel, 3, constrained_layout=True)
                    if channel == 1:
                        axs = axs.reshape(1, 3)
                    for row in range(channel):
                        if hyper_para.dims == 2:
                            im1 = axs[row, 0].imshow(inference[..., row].cpu().detach().numpy(), cmap="jet")
                            im2 = axs[row, 1].imshow(target[..., row].cpu().detach().numpy(), cmap="jet")
                            im3 = axs[row, 2].imshow((inference[..., row] - target[..., row]).cpu().detach().numpy(), cmap="jet")
                            fig.colorbar(im1, ax=axs[row, 0], shrink=0.5)
                            fig.colorbar(im2, ax=axs[row, 1], shrink=0.5)
                            fig.colorbar(im3, ax=axs[row, 2], shrink=0.5)
                            axs[row, 0].axis("off")
                            axs[row, 1].axis("off")
                            axs[row, 2].axis("off")
                        elif hyper_para.dims == 1:
                            im1 = axs[row, 0].plot(inference[..., row].cpu().detach().numpy())
                            im2 = axs[row, 1].plot(target[..., row].cpu().detach().numpy())
                            im3 = axs[row, 2].plot((inference[..., row] - target[..., row]).cpu().detach().numpy())
                        axs[row, 0].set_title(f"epoch {i} prediction")
                        axs[row, 1].set_title(f"epoch {i} target")
                        axs[row, 2].set_title(f"epoch {i} error")
                        
                    run["visualization/training"].append(fig)
                    plt.close(fig)
                

            if i % hyper_para.save_every == 0 or i == start_epoch + hyper_para.epochs-1:    # save every "save_every" epochs or last epoch
                if rank == 0 and i != 0:
                    if world_size > 1:
                        savefn(model.module, latents.module, i, optim_net_dec, optim_states)
                    else:
                        savefn(model, latents, i, optim_net_dec, optim_states)
        if rank == 0:
            run.stop()
            dist.destroy_process_group()

    def save(self, model, latents, epoch, optim_nf, optim_latent):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_states_dict": optim_latent.state_dict(),
                "optim_net_dec_dict": optim_nf.state_dict(),
                "hidden_states": latents.state_dict(),
                "epoch": epoch,
            },
            f"{self.hyper_para.save_path}/checkpoint_{epoch}.pt",
        )

    def load(self, checkpoint_id: int, siren_only=False):

        if checkpoint_id == -1:
            import glob

            checkpoint_list = glob.glob(f"{self.hyper_para.ckpt_loading_path}/checkpoint_*.pt")    # get all the checkpoint files
            checkpoint_list = [int(i.split("_")[-1].split(".")[0]) for i in checkpoint_list]  # get all the checkpoint ids
            try:
                checkpoint_id = max(checkpoint_list)
            except ValueError:
                print(
                    f"*No checkpoint found in {self.hyper_para.ckpt_loading_path}*, starting from scratch"
                )
                return 

        print(
            f"loading checkpoint from {self.hyper_para.ckpt_loading_path}/checkpoint_{checkpoint_id}.pt"
        )

        checkpoint = torch.load(
            f"{self.hyper_para.ckpt_loading_path}/checkpoint_{checkpoint_id}.pt"
        )
        
        self.nf.load_state_dict(checkpoint["model_state_dict"])

        self.start_epoch = checkpoint["epoch"]
        
        self.latent_mean = checkpoint["hidden_states"]["latents"].mean(dim=0)
        self.latent_std = checkpoint["hidden_states"]["latents"].std(dim=0)

        if not siren_only:
            # will update both nd and latents
            if hasattr(self, "N_samples"):
                assert self.N_samples == checkpoint["hidden_states"]["latents"].shape[0]
            else:
                self.N_samples = checkpoint["hidden_states"]["latents"].shape[0]
            self.latents = LatentContainer(
                self.N_samples, self.hyper_para.hidden_size, self.hyper_para.dims, self.hyper_para.lumped_latent
            )
            self.latents.load_state_dict(checkpoint["hidden_states"])
            print("&&& Trained latents loaded")

            self.optim_dict = {
                k: checkpoint[k] for k in ["optim_net_dec_dict", "optim_states_dict"]
            }
            
            return self.nf, self.latents, self.optim_dict, checkpoint["epoch"]

        else:
            # will only update the nf network
            self.optim_dict = {}
            # self.optim_dict["optim_net_dec_dict"] = checkpoint["optim_net_dec_dict"]



if __name__ == "__main__":
    import sys
    import os
    
    hp = ri.basic_input(sys.argv[1])

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = hp.master_port
    # os.environ["MASTER_PORT"] = "12334"
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu_id
    print("available GPU:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name())

    mytrainer = trainer(hyper_para=hp, tag='none', infer_dps=False)
    # mytrainer.load(-1)
    # mytrainer.train()
    mytrainer.load(-1, siren_only=False)
    mytrainer.train()
