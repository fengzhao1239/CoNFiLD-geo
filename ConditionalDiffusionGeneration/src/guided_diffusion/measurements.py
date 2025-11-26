'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from ConditionalNeuralField.cnf.inference_function import pass_through_model_batch
from ConditionalNeuralField.cnf.utils.normalize import Normalizer_ts
from ConditionalNeuralField.cnf.nf_networks import SIRENAutodecoder_film, SIRENAutodecoder_mdf_film
import numpy as np
from einops import rearrange
import h5py
from scipy.stats import qmc
import warnings
from scipy.spatial import cKDTree

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper

def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass


# =====================================================================
# ! Self-defined classes for joint generation of solution and parameter
# =====================================================================

# todo ------------------------- sparse measurement -------------------------

@register_operator(name='cartesian_sparse_measurement')
class CartesianOperatorSM(NonLinearOperator):
    '''for sparse measurement, we only need to query the measured predicted points A*x, and provide the measured data for y'''
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                num_probed,
                user_probed=None
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        # self.query_points = self._gene_random_idx(num_probed)   # todo: specify ur measurement points
        if num_probed == 1:
            query_points = [(32, 28)]
        elif num_probed == 3:
            query_points = [(20, 20), (40, 30), (30, 50)]
        elif num_probed == 5:
            query_points = [(10, 10), (20, 55), (28, 26), (40, 46), (55, 15)]
        else:
            query_points = self._gene_random_idx(num_probed)
            warnings.warn(f"Randomly generated {num_probed} points for measurement, please check the results.")
        self.query_points = query_points if user_probed is None else user_probed
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=5,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <N, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=3>
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        # cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.query_points)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.query_points)}, C), but got {cnf_out_traj.shape}"

        output = cnf_out_traj[..., -1]    # <b, t, N, c=2>
        return output
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _gene_random_idx(self, num_pts, max_range=(63, 63)):
        SAMPLER = qmc.LatinHypercube(d=2, seed=42)
        sample = SAMPLER.random(n=num_pts)  # shape: (num_pts, 2), range: [0, 1)
        scaled = qmc.scale(sample, [0, 0], [max_range[0], max_range[1]])
        idxes = [tuple(map(int, pt)) for pt in scaled]
        return idxes

    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = w = 64
        H = W = 640
        x_coord = np.linspace(0, H, h)
        y_coord = np.linspace(0, W, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
        x_min, x_max = xy_coord.min(), xy_coord.max()
        xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        query_pts = self.query_points    # ur query points
        query_coord = torch.stack([xy_coord[i, j] for (i, j) in query_pts], dim=0).float()    # <N, 2>
        assert query_coord.shape == (len(query_pts), 2), f"Expected query coord shape ({len(query_pts)}, 2), but got {query_coord.shape}"
        return query_coord
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||, and visualizing the true param'''
        data = self._norm_cnf(torch.tensor(self.simdata))    # <t, h, w, c=3>
        # data = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        q_coord = self.query_points
        measurements = torch.stack([torch.stack([data[t, i, j, -1] for (i, j) in q_coord]) for t in range(data.shape[0])])  # <t, N, c=?>
        # assert measurements.shape == (data.shape[0], len(q_coord)), \
        #     f"Expected measurements shape ({data.shape[0]}, {len(q_coord)}), but got {measurements.shape}"
        print(f"Measurement shape: {measurements.shape}")
        return measurements.unsqueeze(0).to(self.device)

@register_operator(name='cartesian_sparse_measurement_all')
class CartesianOperatorSMAll(NonLinearOperator):
    '''for sparse measurement, we only need to query the measured predicted points A*x, and provide the measured data for y'''
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                num_probed,
                user_probed=None,
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        # self.query_points = self._gene_random_idx(num_probed)   # todo: specify ur measurement points
        if num_probed == 1:
            query_points = [(32, 28)]
        elif num_probed == 3:
            query_points = [(20, 20), (40, 30), (30, 50)]
        elif num_probed == 5:
            query_points = [(10, 10), (20, 55), (28, 26), (40, 46), (55, 15)]
        else:
            query_points = self._gene_random_idx(num_probed)
            warnings.warn(f"Randomly generated {num_probed} points for measurement, please check the results.")
        self.query_points = query_points if user_probed is None else user_probed
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=5,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <N, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=3>
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        # cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.query_points)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.query_points)}, C), but got {cnf_out_traj.shape}"

        output = cnf_out_traj[..., :]    # <b, t, N, c=2>
        return output
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _gene_random_idx(self, num_pts, max_range=(63, 63)):
        SAMPLER = qmc.LatinHypercube(d=2, seed=42)
        sample = SAMPLER.random(n=num_pts)  # shape: (num_pts, 2), range: [0, 1)
        scaled = qmc.scale(sample, [0, 0], [max_range[0], max_range[1]])
        idxes = [tuple(map(int, pt)) for pt in scaled]
        return idxes

    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = w = 64
        H = W = 640
        x_coord = np.linspace(0, H, h)
        y_coord = np.linspace(0, W, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
        x_min, x_max = xy_coord.min(), xy_coord.max()
        xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        query_pts = self.query_points    # ur query points
        query_coord = torch.stack([xy_coord[i, j] for (i, j) in query_pts], dim=0).float()    # <N, 2>
        assert query_coord.shape == (len(query_pts), 2), f"Expected query coord shape ({len(query_pts)}, 2), but got {query_coord.shape}"
        return query_coord
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||, and visualizing the true param'''
        data = self._norm_cnf(torch.tensor(self.simdata))    # <t, h, w, c=3>
        # data = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        q_coord = self.query_points
        measurements = torch.stack([torch.stack([data[t, i, j, :] for (i, j) in q_coord]) for t in range(data.shape[0])])  # <t, N, c=?>
        # assert measurements.shape == (data.shape[0], len(q_coord)), \
        #     f"Expected measurements shape ({data.shape[0]}, {len(q_coord)}), but got {measurements.shape}"
        print(f"Measurement shape: {measurements.shape}")
        return measurements.unsqueeze(0).to(self.device)

# =============
# Noway 3D
# =============
@register_operator(name='norway3d_sparse_measurement_all')
class Norway3DOperatorSM(NonLinearOperator):
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                num_probed,
                min_range=(1, 1),
                max_range=(62, 116),
                time_start_idx=0,
                time_end_idx=128,
                time_stride=1,
                noise_level=0.0
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.noise_level = noise_level
        if num_probed == 1:
            self.query_points = [(32, 64)]
        elif num_probed == 3:
            self.query_points = [(20, 20), (40, 30), (30, 50)]
        elif num_probed == (2, 3):
            self.query_points = [(np.int64(20), np.int64(20)), (np.int64(20), np.int64(40)), (np.int64(20), np.int64(96)), (np.int64(42), np.int64(20)), (np.int64(42), np.int64(40)), (np.int64(42), np.int64(96))]
        else:
            self.query_points = self._gene_uniform_idx(num_probed, min_range, max_range)
            warnings.warn(f"Uniformed generated {num_probed} points for measurement, please check the results.")
        
        self.time_start_idx = time_start_idx
        self.time_end_idx = time_end_idx
        self.time_stride = time_stride
        if time_start_idx == 0 and time_end_idx == 128 and time_stride == 1:
            print(f"[**Full case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx == 0 and time_end_idx == 128 and time_stride > 1:
            print(f"[**Sparse case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx > 0 or time_end_idx < 128:
            print(f"[**Half case**] Using the time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        else:
            warnings.warn(f"Unexpected time range or stride: start={time_start_idx}, end={time_end_idx}, stride={time_stride}")
            
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=15,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <N, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, 1 l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=3>
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        # cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.query_points)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.query_points)}, C), but got {cnf_out_traj.shape}"
        
        selected = cnf_out_traj[:, self.time_start_idx:self.time_end_idx:self.time_stride, ...]    # <b, t', N, c>
        selected_noisy = selected + torch.randn_like(selected) * self.noise_level    # add some noise, can be adjusted
        return selected_noisy
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = 64
        w = 118
        x_coord = np.linspace(-1, 1, h)
        y_coord = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
        # x_min, x_max = xy_coord.min(), xy_coord.max()    # ! note this normalization
        # xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        query_pts = self.query_points    # ur query points
        query_coord = torch.stack([xy_coord[i, j] for (i, j) in query_pts], dim=0).float()    # <N, 2>
        assert query_coord.shape == (len(query_pts), 2), f"Expected query coord shape ({len(query_pts)}, 2), but got {query_coord.shape}"
        return query_coord
    
    def _gene_random_idx(self, num_pts, max_range=(63, 117)):
        SAMPLER = qmc.LatinHypercube(d=2, seed=42)
        sample = SAMPLER.random(n=num_pts)  # shape: (num_pts, 2), range: [0, 1)
        scaled = qmc.scale(sample, [0, 0], [max_range[0], max_range[1]])
        idxes = [tuple(map(int, pt)) for pt in scaled]
        return idxes

    def _gene_uniform_idx(self, num_pts, min_range=(1, 1), max_range=(62, 116)):
        """
        Uniformly sample indices in a 2D grid.

        Args:
            num_pts (tuple): (num_h, num_w), number of points to sample along each axis.
            max_range (tuple): (max_h, max_w), max index along each axis.

        Returns:
            List[Tuple[int, int]]: Uniformly sampled integer grid indices.
        """
        if isinstance(num_pts, int):
            num_pts = (num_pts, num_pts)
        num_h, num_w = num_pts
        max_h, max_w = max_range
        min_h, min_w = min_range

        hs = np.linspace(min_h, max_h, num=num_h, dtype=int)
        ws = np.linspace(min_w, max_w, num=num_w, dtype=int)

        mesh_h, mesh_w = np.meshgrid(hs, ws, indexing='ij')  # shape: (num_h, num_w)
        coords = list(zip(mesh_h.flatten(), mesh_w.flatten()))
        return coords
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||'''
        data = self._norm_cnf(torch.tensor(self.simdata))    # ! change
        unnormed_data = torch.tensor(self.simdata, dtype=torch.float32)
        # data = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        q_coord = self.query_points
        measurements = torch.stack([torch.stack([data[t, i, j, :] for (i, j) in q_coord]) for t in range(data.shape[0])])  # <t, N, c=?>
        self.unnormed_measurements = torch.stack([torch.stack([unnormed_data[t, i, j, :] for (i, j) in q_coord]) for t in range(unnormed_data.shape[0])])  # <t, N, c=?>
        print(f"Measurement shape: {measurements.shape}")
        assert measurements.shape == self.unnormed_measurements.shape
        
        slected_measurements = measurements[self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        return slected_measurements.unsqueeze(0).to(self.device)    # <1, t', N, c=3>
    
    def retrieve_data(self):
        retrieved = {
            'simdata': self.simdata,
            'measurement': self.unnormed_measurements,    # <t, N, c=3>
        }
        return retrieved


# =============
# Topography
# =============
@register_operator(name='topography_sparsewell')
class Topography3DOperatorSM(NonLinearOperator):
    
        # --------------5.9 km---------------
        # |                                 |
        # |                                 |
        # 3.2 km                            3.2 km
        # |                                 |
        # |                                 |
        # --------------5.9 km---------------
        
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                query_coords,  # default the production well
                time_start_idx=0,
                time_end_idx=128,
                time_stride=1,
                noise_level=[0., 0., 0., 0.]
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.noise_level = noise_level
        self.query_coords = query_coords
        
        self.time_start_idx = time_start_idx
        self.time_end_idx = time_end_idx
        self.time_stride = time_stride
        if time_start_idx == 0 and time_end_idx == 128 and time_stride == 1:
            print(f"[**Full case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx == 0 and time_end_idx == 128 and time_stride > 1:
            print(f"[**Sparse case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx > 0 or time_end_idx < 128:
            print(f"[**Half case**] Using the time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        else:
            warnings.warn(f"Unexpected time range or stride: start={time_start_idx}, end={time_end_idx}, stride={time_stride}")
            
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=20,
                                                in_coord_features=2,
                                                in_latent_features=384,
                                                out_features=4,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.in_normalizer = Normalizer_ts(method="-11", dim=0)
        self.in_normalizer.params = normalize_records["x_normalizer_params"]
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
        self.coords = np.load('/ehome/zhao/nf/CoNFiLD/Dataset/unstructured_coords.npy')
        self.query_idx = self._get_query_idx()
        print(f'Unstructured coordinates loaded, with shape {self.coords.shape}')
        print(f'>>>>>>> Queried coordinates idxes: {self.query_idx}')
        print(f'>>>>>>>Total query points = {len(self.query_coords)},\n with coordinates: {self._cnf_sparse_in_coord()}')
    
    def _cnf_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self._cnf_all_in_coord().to(self.device)    # <N, coord_dim=2>
        cnf_coord_in = self.in_normalizer.normalize(cnf_coord_in)    # ! norm the coordinates
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=4>
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        return cnf_out_traj, traj_num, time_steps
    
    def _forward_sparse(self, data, **kwargs):
        cnf_out_traj, traj_num, time_steps = self._cnf_forward(data, **kwargs)    # <b, t, N, c>
        cnf_out_traj = cnf_out_traj[:, :, self.query_idx, :]    # <b, t, N_query, c>
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.query_coords)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.query_coords)}, C), but got {cnf_out_traj.shape}"
        
        selected = cnf_out_traj[:, self.time_start_idx:self.time_end_idx:self.time_stride, ...]    # <b, t', N, c>
        return selected[..., :]  # todo we think the sparse measurement is not noisy, can observe all variables
    
    def forward(self, data, **kwargs):
        return self._forward_sparse(data, **kwargs)
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _get_query_idx(self):
        tree = cKDTree(self.coords)
        queries = np.array(self.query_coords)
        assert len(queries.shape) == 2, f"Expected query_coords shape (N, 2), but got {queries.shape}"
        dist, idx = tree.query(queries)
        assert len(idx) == len(queries), f"Expected idx length {len(queries)}, but got {len(idx)}"
        return idx
    
    def _cnf_all_in_coord(self):
        return torch.tensor(self.coords, dtype=torch.float32)    # <N, 2>
    
    def _cnf_sparse_in_coord(self):
        query_idx = self._get_query_idx()    # <7720, 2>  --> <N, 2>
        return torch.tensor(self.coords[query_idx], dtype=torch.float32)    # <N, 2>
    
    def sparse_measurement(self):
        data = self._norm_cnf(torch.tensor(self.simdata))    # ! change
        measurements = data[:, self._get_query_idx(), :]  # <t, N, c=4>
        print(f"@_@ Sparse measurement shape: {measurements.shape}")
        slected_measurements = measurements[self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        return slected_measurements.unsqueeze(0).to(self.device)[..., :]
    
    def measurement(self):
        return self.sparse_measurement()



# todo ------------------------- low fidelity measurement (super-resolution) -------------------------
# =============
# Cartesian
# =============

@register_operator(name='cartesian_superresolution')
class CartesianOperatorSR(NonLinearOperator):
    '''for sparse measurement, we only need to query the measured predicted points A*x, and provide the measured data for y'''
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                ds_size,
                vanilla_flag=False,
                noise=None,
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.ds_size = (ds_size, ds_size) if isinstance(ds_size, int) else tuple(ds_size)    # todo: specify ur downsampling size
        self.vanilla_flag = vanilla_flag
        if vanilla_flag:
            warnings.warn(f"Vanilla flag is set to True, using the original data without downsampling!")
        if noise is not None:
            warnings.warn(f"Noise is set to {noise}, will add noise to the output data.")
            self.noise = noise
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=5,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <h, w, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1).unsqueeze(1)    # <(b*t), 1, 1 l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, h, w, 2>
        assert len(cnf_latent_in.shape) == len(cnf_coord_in.shape) == 4, \
            f"CNF Decoder input shape [{cnf_coord_in.shape}, {cnf_latent_in.shape}] mismatch, expected 4D tensor"
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), h, w, c=3>
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) h w c-> b t h w c", b=traj_num, t=time_steps)    # <b, t, h, w, c>
        # cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:2] == (traj_num, time_steps), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, H, W, C), but got {cnf_out_traj.shape}"
        
        # * downsampling
        cnf_out_traj_reshape = rearrange(cnf_out_traj, "b t h w c -> (b t) c h w")
        if self.vanilla_flag:
            down_cnf_out_traj = cnf_out_traj_reshape
        else:
            down_cnf_out_traj = F.interpolate(cnf_out_traj_reshape, size=self.ds_size, mode='nearest')
        output = rearrange(down_cnf_out_traj, "(b t) c h w -> b t h w c", b=traj_num, t=time_steps)
        # 0:1 -- pressure
        # 1:2 -- saturation
        # 2:3 -- permeability
        return output[:, :, ..., -1]    # <b, t, h, w, c=1>
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = w = 64
        H = W = 640
        x_coord = np.linspace(0, H, h)
        y_coord = np.linspace(0, W, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
        x_min, x_max = xy_coord.min(), xy_coord.max()
        xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        return xy_coord
    
    def _add_channelwise_noise(self, data, noise_levels):
        noise_levels = torch.tensor(noise_levels, dtype=data.dtype).to(self.device)
        noise = torch.randn_like(data) * noise_levels
        mask = (data != 0).float()
        return data + noise * mask
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||'''
        # data = self.simdata   # <t, h, w, c=3>
        # data = self._norm_cnf(torch.tensor(self.simdata))    # <t, h, w, c=3>
        data = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        if hasattr(self, 'noise') and self.noise is not None:
            print(f"Adding noise with levels {self.noise} to the data.")
            data = self._add_channelwise_noise(data, self.noise)
        data_reshape = rearrange(data, "t h w c -> t c h w")
        if self.vanilla_flag:
            ds_measurements = data_reshape
        else:
            ds_measurements = F.interpolate(data_reshape, size=self.ds_size, mode='nearest')    # <t, c, ds_h, ds_w>
        out_measurements = rearrange(ds_measurements, "t c h w -> t h w c")
        out_measurements = self._norm_cnf(out_measurements)
        return out_measurements.unsqueeze(0).to(self.device)[:, :, ..., -1]    # <1, t, h, w, c=1>
    
# =============
# Noway 3D
# =============
@register_operator(name='norway3d_superresolution')
class Norway3DOperatorSR(NonLinearOperator):
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                ds_size,
                vanilla_flag=False,
                noise_level=0.0,
                time_start_idx=0,
                time_end_idx=128,
                time_stride=1,
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.noise_level = noise_level
        assert isinstance(ds_size, tuple), f"ds_size should be tuple, but got {type(ds_size)}"
        self.ds_size = ds_size    # todo: specify ur downsampling size
        self.vanilla_flag = vanilla_flag
        if vanilla_flag:
            warnings.warn(f"Vanilla flag is set to True, using the original data without downsampling!")
        
        self.time_start_idx = time_start_idx
        self.time_end_idx = time_end_idx
        self.time_stride = time_stride
        if time_start_idx == 0 and time_end_idx == 128 and time_stride == 1:
            print(f"[**Full case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx == 0 and time_end_idx == 128 and time_stride > 1:
            print(f"[**Sparse case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx > 0 or time_end_idx < 128:
            print(f"[**Half case**] Using the time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        else:
            warnings.warn(f"Unexpected time range or stride: start={time_start_idx}, end={time_end_idx}, stride={time_stride}")
            
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=15,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <h, w, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1).unsqueeze(1)    # <(b*t), 1, 1 l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, h, w, 2>
        assert len(cnf_latent_in.shape) == len(cnf_coord_in.shape) == 4, \
            f"CNF Decoder input shape [{cnf_coord_in.shape}, {cnf_latent_in.shape}] mismatch, expected 4D tensor"
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), h, w, c=3>
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) h w c-> b t h w c", b=traj_num, t=time_steps)    # <b, t, h, w, c>
        cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:2] == (traj_num, time_steps), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, H, W, C), but got {cnf_out_traj.shape}"
        
        # * downsampling
        cnf_out_traj_reshape = rearrange(cnf_out_traj, "b t h w c -> (b t) c h w")
        if self.vanilla_flag:
            down_cnf_out_traj = cnf_out_traj_reshape
        else:
            down_cnf_out_traj = F.interpolate(cnf_out_traj_reshape, size=self.ds_size, mode='nearest')
        output = rearrange(down_cnf_out_traj, "(b t) c h w -> b t h w c", b=traj_num, t=time_steps)
        # * 0:1 -- pressure
        # * 1:2 -- saturation
        # * 2:3 -- permeability
        sat = output[:, :, ..., 1:2]    # <b, t, h, w, c=1>
        sat = torch.where(sat < 0.02, torch.zeros_like(sat), sat)  # filter out values smaller than 0.02
        mask = (sat != 0).float()
        noise = torch.randn_like(sat) * self.noise_level    # generate noise with the same shape as sat
        noisy_sat = sat + noise * mask    # add noise only to non-zero grids
        selected = noisy_sat[:, self.time_start_idx:self.time_end_idx:self.time_stride, :, :, :]  # <b, t', h, w, c=1>
        return selected
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = 64
        w = 118
        x_coord = np.linspace(-1, 1, h)
        y_coord = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
        # x_min, x_max = xy_coord.min(), xy_coord.max()    # ! note this normalization
        # xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        return xy_coord
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||'''
        # data = self._norm_cnf(torch.tensor(self.simdata))    # ! change
        data = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        h, w = data.shape[1], data.shape[2]
        data_reshape = rearrange(data, "t h w c -> t c h w")
        if self.vanilla_flag:
            ds_measurements = data_reshape
            up_measurements = data_reshape
        else:
            ds_measurements = F.interpolate(data_reshape, size=self.ds_size, mode='nearest')    # <t, c, ds_h, ds_w>
            up_measurements = F.interpolate(ds_measurements, size=(h, w), mode='nearest')    # * upsample for visualization
        out_measurements = rearrange(ds_measurements, "t c h w -> t h w c")
        up_out_measurements = rearrange(up_measurements, "t c h w -> t h w c")
        
        self.unnormed_measurments = out_measurements    # <t, h, w, c=3>
        self.unnormed_up_measurments = up_out_measurements    # <t, h, w, c=3>
        
        final_out = out_measurements.unsqueeze(0).to(self.device)[:, :, ..., 1:2]    # <1, t, h, w, c=1>
        slected_final_out = final_out[:, self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        return slected_final_out
    
    def retrieve_data(self):
        retrieved = {
            'simdata': self.simdata,
            'measurement': self.unnormed_measurments.detach().cpu().numpy(),
            'up_measurement': self.unnormed_up_measurments.detach().cpu().numpy(),
        }
        return retrieved
    
# =============
# Topography
# =============
@register_operator(name='topography_geoseismic')
class Topography3DOperatorGEODATA(NonLinearOperator):
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                time_start_idx=0,
                time_end_idx=128,
                time_stride=1,
                noise_level=[0., 0., 0., 0.]
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.noise_level = noise_level
        
        self.time_start_idx = time_start_idx
        self.time_end_idx = time_end_idx
        self.time_stride = time_stride
        if time_start_idx == 0 and time_end_idx == 128 and time_stride == 1:
            print(f"[**Full case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx == 0 and time_end_idx == 128 and time_stride > 1:
            print(f"[**Sparse case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx > 0 or time_end_idx < 128:
            print(f"[**Half case**] Using the time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        else:
            warnings.warn(f"Unexpected time range or stride: start={time_start_idx}, end={time_end_idx}, stride={time_stride}")
            
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=20,
                                                in_coord_features=2,
                                                in_latent_features=384,
                                                out_features=4,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.in_normalizer = Normalizer_ts(method="-11", dim=0)
        self.in_normalizer.params = normalize_records["x_normalizer_params"]
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
        self.coords = np.load('/ehome/zhao/nf/CoNFiLD/Dataset/unstructured_coords.npy')
        print(f'Unstructured coordinates loaded, with shape {self.coords.shape}')
    
    def _cnf_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self._cnf_all_in_coord().to(self.device)    # <N, coord_dim=2>
        cnf_coord_in = self.in_normalizer.normalize(cnf_coord_in)    # ! norm the coordinates
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=4>
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        return cnf_out_traj, traj_num, time_steps
    
    def _forward_field_1(self, data, **kwargs):
        cnf_out_traj, traj_num, time_steps = self._cnf_forward(data, **kwargs)    # <b, t, N, c>
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.coords)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.coords)}, C), but got {cnf_out_traj.shape}"
        selected = cnf_out_traj[:, self.time_start_idx:self.time_end_idx:self.time_stride, :, :]    # <b, t', N, c>
        return selected[..., 2:]  # todo  geodata
    
    def forward(self, data, **kwargs):
        return self._forward_field_1(data, **kwargs)
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _cnf_all_in_coord(self):
        return torch.tensor(self.coords, dtype=torch.float32)    # <N, 2>
    
    def _add_channelwise_noise(self, data, noise_levels):
        noise_levels = torch.tensor(noise_levels, dtype=data.dtype).to(self.device)
        noise = torch.randn_like(data) * noise_levels
        return data + noise
    
    def field_measurement_1(self):
        '''will be used for || y - A * x ||'''
        measurements = torch.tensor(self.simdata, device=self.device)    # ! change
        # measurements = data[:, :, 2:]  # <t, N, c=4>  # todo geodata
        print(f"@_@ Field measurement shape: {measurements.shape}")
        slected_measurements = measurements[self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        if np.array(self.noise_level).any() > 0:
            print(f"@_@ Adding noise to the field measurement with levels: {self.noise_level}")
            selected_noisy = self._add_channelwise_noise(slected_measurements, self.noise_level)    # add some noise, can be adjusted
            selected_noisy = self._norm_cnf(selected_noisy)    # norm the noisy data, again
        else:
            selected_noisy = self._norm_cnf(slected_measurements)
        return selected_noisy.unsqueeze(0).to(self.device)[..., 2:]
    
    def measurement(self):
        return self.field_measurement_1()


@register_operator(name='topography_co2seismic')
class Topography3DOperatorCO2DATA(NonLinearOperator):
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                time_start_idx=0,
                time_end_idx=128,
                time_stride=1,
                noise_level=[0., 0., 0., 0.]
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.noise_level = noise_level
        
        self.time_start_idx = time_start_idx
        self.time_end_idx = time_end_idx
        self.time_stride = time_stride
        if time_start_idx == 0 and time_end_idx == 128 and time_stride == 1:
            print(f"[**Full case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx == 0 and time_end_idx == 128 and time_stride > 1:
            print(f"[**Sparse case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx > 0 or time_end_idx < 128:
            print(f"[**Half case**] Using the time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        else:
            warnings.warn(f"Unexpected time range or stride: start={time_start_idx}, end={time_end_idx}, stride={time_stride}")
            
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=20,
                                                in_coord_features=2,
                                                in_latent_features=384,
                                                out_features=4,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.in_normalizer = Normalizer_ts(method="-11", dim=0)
        self.in_normalizer.params = normalize_records["x_normalizer_params"]
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
        self.coords = np.load('/ehome/zhao/nf/CoNFiLD/Dataset/unstructured_coords.npy')
        print(f'Unstructured coordinates loaded, with shape {self.coords.shape}')
    
    def _cnf_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self._cnf_all_in_coord().to(self.device)    # <N, coord_dim=2>
        cnf_coord_in = self.in_normalizer.normalize(cnf_coord_in)    # ! norm the coordinates
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=4>
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        return cnf_out_traj, traj_num, time_steps
    
    def _forward_field_0(self, data, **kwargs):
        cnf_out_traj, traj_num, time_steps = self._cnf_forward(data, **kwargs)    # <b, t, N, c>
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.coords)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.coords)}, C), but got {cnf_out_traj.shape}"
        selected = cnf_out_traj[:, self.time_start_idx:self.time_end_idx:self.time_stride, :, :]    # <b, t', N, c>
        selected = self._unnorm_cnf(selected)    # unnorm the selected data
        return selected[..., 1]  # todo  co2 saturation
    
    def forward(self, data, **kwargs):
        return self._forward_field_0(data, **kwargs)
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _cnf_all_in_coord(self):
        return torch.tensor(self.coords, dtype=torch.float32)    # <N, 2>
        
    def field_measurement_0(self):
        '''will be used for || y - A * x ||'''
        data = torch.tensor(self.simdata)    # ! change
        measurements = data[:, :, 1]  # <t, N, c=4>  # todo co2 saturation
        print(f"@_@ Field measurement shape: {measurements.shape}")
        slected_measurements = measurements[self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        if self.noise_level[1] > 0:
            print(f"@_@ Adding noise to the field measurement with levels: {self.noise_level[1]}")
            sat = slected_measurements
            mask = (sat != 0).float()
            noise = self.noise_level[1] * torch.randn_like(sat)
            noisy_sat = sat + noise * mask
        else:
            noisy_sat = slected_measurements
        return noisy_sat.unsqueeze(0).to(self.device)
    
    def measurement(self):
        return self.field_measurement_0()

# todo ------------------------- damaged measurement -------------------------

@register_operator(name='cartesian_inpainting')
class CartesianOperatorIP_v2(NonLinearOperator):
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                damage_size=16,
                state_variable='saturation',    # todo: specify ur state variable
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.damage_size = damage_size if isinstance(damage_size, tuple) else (damage_size, damage_size)    # todo: specify ur damage size
        # ------ mask -------
        h, w = 64, 64
        mask = torch.ones((h, w), dtype=torch.float32, device=device)
        start_h = (h - self.damage_size[0]) // 2
        start_w = (w - self.damage_size[1]) // 2
        mask[start_h:start_h + self.damage_size[0], start_w:start_w + self.damage_size[1]] = 0.0
        self.mask = mask
        # -------------------
        if state_variable == 'saturation':
            self.select_dim = 1
        elif state_variable == 'pressure':
            self.select_dim = 0
        elif state_variable == 'permeability':
            self.select_dim = 2
        else:
            raise ValueError(f"Unknown state variable: {state_variable}. Supported: 'saturation', 'pressure', 'permeability'.")
        
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=5,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <N, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=3>
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        # cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:2] == (traj_num, time_steps), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, N, C), but got {cnf_out_traj.shape}"
        # 0:1 -- pressure
        # 1:2 -- saturation
        # 2:3 -- permeability
        output = cnf_out_traj[:, :, ..., self.select_dim]    # <b, t, N>
        return output
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = w = 64
        H = W = 640
        x_coord = np.linspace(0, H, h)
        y_coord = np.linspace(0, W, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32, device=self.device)
        x_min, x_max = xy_coord.min(), xy_coord.max()
        xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        query_coord = xy_coord[self.mask.bool()]
        # print(f"Query coordinates shape: {query_coord.shape}")  # <N, 2>
        # print(f"Query coordinates: {query_coord}")
        return query_coord  # <N, 2>
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||'''
        # data = self.simdata   # <t, h, w, c=3>
        data = self._norm_cnf(torch.tensor(self.simdata)).to(self.device)    # <t, h, w, c=3>
        i_idx, j_idx  = self.mask.bool().nonzero(as_tuple=True)
        measurements = data[:, i_idx, j_idx, :]  # <t, N, 3>
        print(f"Measurements shape: {measurements.shape}")  # <t, N, c=3>
        out_measurements = measurements[..., self.select_dim].unsqueeze(0)    # <1, t, N>
        return out_measurements
    
    def retrieve_data(self):
        retrieved = {
            'simdata': self.simdata,
            'measurement': self.simdata[..., self.select_dim] * self.mask.unsqueeze(0).detach().cpu().numpy(),    # <t, h, w>
            'mask': self.mask.detach().cpu().numpy(),    # <h, w>
        }
        return retrieved

@register_operator(name='cartesian_inpainting')
class CartesianOperatorIP(NonLinearOperator):
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                damage_size=16,
                state_variable='saturation',    # todo: specify ur state variable
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.damage_size = damage_size if isinstance(damage_size, tuple) else (damage_size, damage_size)    # todo: specify ur damage size
        # ------ mask -------
        h, w = 64, 64
        mask = torch.ones((h, w), dtype=torch.float32, device=device)
        start_h = (h - self.damage_size[0]) // 2
        start_w = (w - self.damage_size[1]) // 2
        mask[start_h:start_h + self.damage_size[0], start_w:start_w + self.damage_size[1]] = 0.0
        self.mask = mask.unsqueeze(0).unsqueeze(0)    # <1, 1, h, w>
        # -------------------
        if state_variable == 'saturation':
            self.select_dim = 1
        elif state_variable == 'pressure':
            self.select_dim = 0
        elif state_variable == 'permeability':
            self.select_dim = 2
        else:
            raise ValueError(f"Unknown state variable: {state_variable}. Supported: 'saturation', 'pressure', 'permeability'.")
        
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=5,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <h, w, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1).unsqueeze(1)    # <(b*t), 1, 1 l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, h, w, 2>
        assert len(cnf_latent_in.shape) == len(cnf_coord_in.shape) == 4, \
            f"CNF Decoder input shape [{cnf_coord_in.shape}, {cnf_latent_in.shape}] mismatch, expected 4D tensor"
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), h, w, c=3>
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) h w c-> b t h w c", b=traj_num, t=time_steps)    # <b, t, h, w, c>
        # cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:2] == (traj_num, time_steps), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, H, W, C), but got {cnf_out_traj.shape}"
        # 0:1 -- pressure
        # 1:2 -- saturation
        # 2:3 -- permeability
        selected = cnf_out_traj[:, :, ..., self.select_dim]    # <b, t, h, w>
        output = self.mask * selected
        return output
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = w = 64
        H = W = 640
        x_coord = np.linspace(0, H, h)
        y_coord = np.linspace(0, W, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
        x_min, x_max = xy_coord.min(), xy_coord.max()
        xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        return xy_coord
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||'''
        # data = self.simdata   # <t, h, w, c=3>
        data = self._norm_cnf(torch.tensor(self.simdata)).unsqueeze(0).to(self.device)    # <1, t, h, w, c=3>
        selected = data[:, :, ..., self.select_dim]    # <1, t, h, w>
        out_measurements = self.mask * selected    # <1, t, h, w>
        return out_measurements
    
    def retrieve_data(self):
        retrieved = {
            'simdata': self.simdata,
            'measurement': self.simdata[..., self.select_dim] * self.mask.squeeze(0).detach().cpu().numpy(),    # <t, h, w>
            'mask': self.mask.squeeze().detach().cpu().numpy(),    # <h, w>
        }
        return retrieved


# todo ------------------------- Multi source measurement -------------------------
@register_operator(name='topography_sparsewell_and_co2seismic')
class Topography3DOperatorCASEONE(NonLinearOperator):
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                query_coords=[[1650.208, 4182.751]],  # default the production well
                time_start_idx=0,
                time_end_idx=128,
                time_stride=1,
                noise_level=[0., 0., 0., 0.]
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.noise_level = noise_level
        self.query_coords = query_coords
        
        self.time_start_idx = time_start_idx
        self.time_end_idx = time_end_idx
        self.time_stride = time_stride
        if time_start_idx == 0 and time_end_idx == 128 and time_stride == 1:
            print(f"[**Full case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx == 0 and time_end_idx == 128 and time_stride > 1:
            print(f"[**Sparse case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx > 0 or time_end_idx < 128:
            print(f"[**Half case**] Using the time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        else:
            warnings.warn(f"Unexpected time range or stride: start={time_start_idx}, end={time_end_idx}, stride={time_stride}")
            
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=20,
                                                in_coord_features=2,
                                                in_latent_features=384,
                                                out_features=4,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.in_normalizer = Normalizer_ts(method="-11", dim=0)
        self.in_normalizer.params = normalize_records["x_normalizer_params"]
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
        self.coords = np.load('/ehome/zhao/nf/CoNFiLD/Dataset/unstructured_coords.npy')
        self.query_idx = self._get_query_idx()
        print(f'Unstructured coordinates loaded, with shape {self.coords.shape}')
        print(f'>>>>>>> Queried coordinates idxes: {self.query_idx}')
        print(f'>>>>>>>Total query points = {len(self.query_coords)},\n with coordinates: {self._cnf_sparse_in_coord()}')
    
    def _cnf_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self._cnf_all_in_coord().to(self.device)    # <N, coord_dim=2>
        cnf_coord_in = self.in_normalizer.normalize(cnf_coord_in)    # ! norm the coordinates
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=4>
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        return cnf_out_traj, traj_num, time_steps
    
    def _forward_field(self, data, **kwargs):
        cnf_out_traj, traj_num, time_steps = self._cnf_forward(data, **kwargs)    # <b, t, N, c>
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.coords)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.coords)}, C), but got {cnf_out_traj.shape}"
        selected = cnf_out_traj[:, self.time_start_idx:self.time_end_idx:self.time_stride, :, :]    # <b, t', N, c>
        selected = self._unnorm_cnf(selected)    # unnorm the selected data
        return selected[..., 1]  # todo  co2 saturation
    
    def _forward_sparse(self, data, **kwargs):
        cnf_out_traj, traj_num, time_steps = self._cnf_forward(data, **kwargs)    # <b, t, N, c>
        cnf_out_traj = cnf_out_traj[:, :, self.query_idx, :]    # <b, t, N_query, c>
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.query_coords)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.query_coords)}, C), but got {cnf_out_traj.shape}"
        
        selected = cnf_out_traj[:, self.time_start_idx:self.time_end_idx:self.time_stride, ...]    # <b, t', N, c>
        return selected[..., :]  # todo we think the sparse measurement is not noisy, can observe all variables
    
    def forward(self, data, **kwargs):
        return self._forward_field(data, **kwargs), self._forward_sparse(data, **kwargs)
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _get_query_idx(self):
        tree = cKDTree(self.coords)
        queries = np.array(self.query_coords)
        assert len(queries.shape) == 2, f"Expected query_coords shape (N, 2), but got {queries.shape}"
        dist, idx = tree.query(queries)
        assert len(idx) == len(queries), f"Expected idx length {len(queries)}, but got {len(idx)}"
        return idx
    
    def _cnf_all_in_coord(self):
        return torch.tensor(self.coords, dtype=torch.float32)    # <N, 2>
    
    def _cnf_sparse_in_coord(self):
        query_idx = self._get_query_idx()    # <7720, 2>  --> <N, 2>
        return torch.tensor(self.coords[query_idx], dtype=torch.float32)    # <N, 2>
    
    def _add_channelwise_noise(self, data, noise_levels):
        noise_levels = torch.tensor(noise_levels, dtype=data.dtype).to(self.device)
        noise = torch.randn_like(data) * noise_levels
        return data + noise
        
    def field_measurement(self):
        '''will be used for || y - A * x ||'''
        data = torch.tensor(self.simdata)    # ! change
        measurements = data[:, :, 1]  # <t, N, c=4>  # todo co2 saturation
        print(f"@_@ Field measurement shape: {measurements.shape}")
        slected_measurements = measurements[self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        if self.noise_level[1] > 0:
            print(f"@_@ Adding noise to the field measurement with levels: {self.noise_level[1]}")
            sat = slected_measurements
            mask = (sat != 0).float()
            noise = self.noise_level[1] * torch.randn_like(sat)
            noisy_sat = sat + noise * mask
        else:
            noisy_sat = slected_measurements
        return noisy_sat.unsqueeze(0).to(self.device)
    
    def sparse_measurement(self):
        data = self._norm_cnf(torch.tensor(self.simdata))    # ! change
        measurements = data[:, self._get_query_idx(), :]  # <t, N, c=4>
        print(f"@_@ Sparse measurement shape: {measurements.shape}")
        slected_measurements = measurements[self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        return slected_measurements.unsqueeze(0).to(self.device)[..., :]
    
    def measurement(self):
        return self.field_measurement(), self.sparse_measurement()


@register_operator(name='topography_sparsewell_and_geoseismic')
class Topography3DOperatorCASETWO(NonLinearOperator):
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                query_coords,  # default the production well
                time_start_idx=0,
                time_end_idx=128,
                time_stride=1,
                noise_level=[0., 0., 0., 0.]
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.noise_level = noise_level
        self.query_coords = query_coords
        
        self.time_start_idx = time_start_idx
        self.time_end_idx = time_end_idx
        self.time_stride = time_stride
        if time_start_idx == 0 and time_end_idx == 128 and time_stride == 1:
            print(f"[**Full case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx == 0 and time_end_idx == 128 and time_stride > 1:
            print(f"[**Sparse case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx > 0 or time_end_idx < 128:
            print(f"[**Half case**] Using the time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        else:
            warnings.warn(f"Unexpected time range or stride: start={time_start_idx}, end={time_end_idx}, stride={time_stride}")
            
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=20,
                                                in_coord_features=2,
                                                in_latent_features=384,
                                                out_features=4,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.in_normalizer = Normalizer_ts(method="-11", dim=0)
        self.in_normalizer.params = normalize_records["x_normalizer_params"]
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
        self.coords = np.load('/ehome/zhao/nf/CoNFiLD/Dataset/unstructured_coords.npy')
        self.query_idx = self._get_query_idx()
        print(f'Unstructured coordinates loaded, with shape {self.coords.shape}')
        print(f'>>>>>>> Queried coordinates idxes: {self.query_idx}')
        print(f'>>>>>>>Total query points = {len(self.query_coords)},\n with coordinates: {self._cnf_sparse_in_coord()}')
    
    def _cnf_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self._cnf_all_in_coord().to(self.device)    # <N, coord_dim=2>
        cnf_coord_in = self.in_normalizer.normalize(cnf_coord_in)    # ! norm the coordinates
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=4>
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        return cnf_out_traj, traj_num, time_steps
    
    def _forward_field(self, data, **kwargs):
        cnf_out_traj, traj_num, time_steps = self._cnf_forward(data, **kwargs)    # <b, t, N, c>
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.coords)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.coords)}, C), but got {cnf_out_traj.shape}"
        selected = cnf_out_traj[:, self.time_start_idx:self.time_end_idx:self.time_stride, :, :]    # <b, t', N, c>
        return selected[..., 2:]  # todo  geodata
    
    def _forward_sparse(self, data, **kwargs):
        cnf_out_traj, traj_num, time_steps = self._cnf_forward(data, **kwargs)    # <b, t, N, c>
        cnf_out_traj = cnf_out_traj[:, :, self.query_idx, :]    # <b, t, N_query, c>
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.query_coords)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.query_coords)}, C), but got {cnf_out_traj.shape}"
        
        selected = cnf_out_traj[:, self.time_start_idx:self.time_end_idx:self.time_stride, ...]    # <b, t', N, c>
        return selected[..., :]  # todo we think the sparse measurement is not noisy, can observe all variables
    
    def forward(self, data, **kwargs):
        return self._forward_field(data, **kwargs), self._forward_sparse(data, **kwargs)
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _get_query_idx(self):
        tree = cKDTree(self.coords)
        queries = np.array(self.query_coords)
        assert len(queries.shape) == 2, f"Expected query_coords shape (N, 2), but got {queries.shape}"
        dist, idx = tree.query(queries)
        assert len(idx) == len(queries), f"Expected idx length {len(queries)}, but got {len(idx)}"
        return idx
    
    def _cnf_all_in_coord(self):
        return torch.tensor(self.coords, dtype=torch.float32)    # <N, 2>
    
    def _cnf_sparse_in_coord(self):
        query_idx = self._get_query_idx()    # <7720, 2>  --> <N, 2>
        return torch.tensor(self.coords[query_idx], dtype=torch.float32)    # <N, 2>
    
    def _add_channelwise_noise(self, data, noise_levels):
        noise_levels = torch.tensor(noise_levels, dtype=data.dtype).to(self.device)
        noise = torch.randn_like(data) * noise_levels
        return data + noise
        
    def field_measurement(self):
        '''will be used for || y - A * x ||'''
        measurements = torch.tensor(self.simdata, device=self.device)    # ! change
        # measurements = data[:, :, 2:]  # <t, N, c=4>  # todo geodata
        print(f"@_@ Field measurement shape: {measurements.shape}")
        slected_measurements = measurements[self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        if np.array(self.noise_level).any() > 0:
            print(f"@_@ Adding noise to the field measurement with levels: {self.noise_level}")
            selected_noisy = self._add_channelwise_noise(slected_measurements, self.noise_level)    # add some noise, can be adjusted
            selected_noisy = self._norm_cnf(selected_noisy)    # norm the noisy data, again
        else:
            selected_noisy = self._norm_cnf(slected_measurements)
        return selected_noisy.unsqueeze(0).to(self.device)[..., 2:]
    
    def sparse_measurement(self):
        data = self._norm_cnf(torch.tensor(self.simdata))    # ! change
        measurements = data[:, self._get_query_idx(), :]  # <t, N, c=4>
        print(f"@_@ Sparse measurement shape: {measurements.shape}")
        slected_measurements = measurements[self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        return slected_measurements.unsqueeze(0).to(self.device)[..., :]
    
    def measurement(self):
        return self.field_measurement(), self.sparse_measurement()


@register_operator(name='topography_co2seismic_and_geoseismic')
class Topography3DOperatorCASETHREE(NonLinearOperator):
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                time_start_idx=0,
                time_end_idx=128,
                time_stride=1,
                noise_level=[0., 0., 0., 0.]
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.noise_level = noise_level
        
        self.time_start_idx = time_start_idx
        self.time_end_idx = time_end_idx
        self.time_stride = time_stride
        if time_start_idx == 0 and time_end_idx == 128 and time_stride == 1:
            print(f"[**Full case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx == 0 and time_end_idx == 128 and time_stride > 1:
            print(f"[**Sparse case**] Using the full time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        elif time_start_idx > 0 or time_end_idx < 128:
            print(f"[**Half case**] Using the time range from {time_start_idx} to {time_end_idx} with stride {time_stride}.")
        else:
            warnings.warn(f"Unexpected time range or stride: start={time_start_idx}, end={time_end_idx}, stride={time_stride}")
            
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=20,
                                                in_coord_features=2,
                                                in_latent_features=384,
                                                out_features=4,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.in_normalizer = Normalizer_ts(method="-11", dim=0)
        self.in_normalizer.params = normalize_records["x_normalizer_params"]
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
        self.coords = np.load('/ehome/zhao/nf/CoNFiLD/Dataset/unstructured_coords.npy')
        print(f'Unstructured coordinates loaded, with shape {self.coords.shape}')
    
    def _cnf_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self._cnf_all_in_coord().to(self.device)    # <N, coord_dim=2>
        cnf_coord_in = self.in_normalizer.normalize(cnf_coord_in)    # ! norm the coordinates
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=4>
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        return cnf_out_traj, traj_num, time_steps
    
    def _forward_field_0(self, data, **kwargs):
        cnf_out_traj, traj_num, time_steps = self._cnf_forward(data, **kwargs)    # <b, t, N, c>
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.coords)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.coords)}, C), but got {cnf_out_traj.shape}"
        selected = cnf_out_traj[:, self.time_start_idx:self.time_end_idx:self.time_stride, :, :]    # <b, t', N, c>
        selected = self._unnorm_cnf(selected)    # unnorm the selected data
        return selected[..., 1]  # todo  co2 saturation
    
    def _forward_field_1(self, data, **kwargs):
        cnf_out_traj, traj_num, time_steps = self._cnf_forward(data, **kwargs)    # <b, t, N, c>
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.coords)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.coords)}, C), but got {cnf_out_traj.shape}"
        selected = cnf_out_traj[:, self.time_start_idx:self.time_end_idx:self.time_stride, :, :]    # <b, t', N, c>
        return selected[..., 2:]  # todo  geodata
    
    def forward(self, data, **kwargs):
        return self._forward_field_0(data, **kwargs), self._forward_field_1(data, **kwargs)
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _cnf_all_in_coord(self):
        return torch.tensor(self.coords, dtype=torch.float32)    # <N, 2>
    
    def _add_channelwise_noise(self, data, noise_levels):
        noise_levels = torch.tensor(noise_levels, dtype=data.dtype).to(self.device)
        noise = torch.randn_like(data) * noise_levels
        return data + noise
        
    def field_measurement_0(self):
        '''will be used for || y - A * x ||'''
        data = torch.tensor(self.simdata)    # ! change
        measurements = data[:, :, 1]  # <t, N, c=4>  # todo co2 saturation
        print(f"@_@ Field measurement shape: {measurements.shape}")
        slected_measurements = measurements[self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        if self.noise_level[1] > 0:
            print(f"@_@ Adding noise to the field measurement with levels: {self.noise_level[1]}")
            sat = slected_measurements
            mask = (sat != 0).float()
            noise = self.noise_level[1] * torch.randn_like(sat)
            noisy_sat = sat + noise * mask
        else:
            noisy_sat = slected_measurements
        return noisy_sat.unsqueeze(0).to(self.device)
    
    def field_measurement_1(self):
        '''will be used for || y - A * x ||'''
        measurements = torch.tensor(self.simdata, device=self.device)    # ! change
        # measurements = data[:, :, 2:]  # <t, N, c=4>  # todo geodata
        print(f"@_@ Field measurement shape: {measurements.shape}")
        slected_measurements = measurements[self.time_start_idx:self.time_end_idx:self.time_stride, ...]
        if np.array(self.noise_level).any() > 0:
            print(f"@_@ Adding noise to the field measurement with levels: {self.noise_level}")
            selected_noisy = self._add_channelwise_noise(slected_measurements, self.noise_level)    # add some noise, can be adjusted
            selected_noisy = self._norm_cnf(selected_noisy)    # norm the noisy data, again
        else:
            selected_noisy = self._norm_cnf(slected_measurements)
        return selected_noisy.unsqueeze(0).to(self.device)[..., 2:]
    
    def measurement(self):
        return self.field_measurement_0(), self.field_measurement_1()
    










# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)