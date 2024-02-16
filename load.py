import numpy as np
from gwinferno.preprocess.data_collection import load_posterior_samples
from utils import chirp_mass_from_m1q


# downsample for ensembling; TODO: replace with masking?
def load_posteriors(dir, use_chirp_mass=True, no_downsample=False, params=('mass_1','mass_ratio','a_1','a_2','cos_tilt_1','cos_tilt_2','redshift','prior')):
    in_posts, names = load_posterior_samples(dir, spin=True, no_downsample=no_downsample)
    posts = {}
    for event in names:
        post = {p: np.array(in_posts[event][p]) for p in params}
        if use_chirp_mass:
            m1 = post.pop("mass_1")
            post['chirp_mass'] = chirp_mass_from_m1q(m1, post['mass_ratio'])
        posts[event] = post
    return posts, names
