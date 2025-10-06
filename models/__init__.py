from .decoder import build
# from .SFA import SFA_Block, SFA_Stage
# from .patch_embed import PatchEmbed
# from .downsample import DownSample, DownSample_CP
# from .gradient_utils import get_gradient_prior
# from .SFPM import Spatial_Fourier_Parallel_Mixer
# from .PFGM import Prior_Feed_Gate_Mixer

# __all__ = [
#     'SFA_Block', 'SFA_Stage',
#     'PatchEmbed', 'DownSample', 'DownSample_CP',
#     'get_gradient_prior',
#     'Spatial_Fourier_Parallel_Mixer',
#     'Prior_Feed_Gate_Mixer'
# ]

def build_model(args):
    return build(args)