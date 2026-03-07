import torch
from torch import nn
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

class VAE(nn.Module):
    def __init__(self, nn_version='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        local_nn = getattr(nns, nn_version)
        self.enc = local_nn.Encoder(self.z_dim)  # from ./nns/vx.py
        self.dec = local_nn.Decoder(self.z_dim)  # from ./nns/vx.py

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl, rec
        ################################################################################
        ### START CODE HERE ###
        m, v = self.enc(x)
        z = ut.sample_gaussian(m, v)
        logits = self.dec(z)
        recon = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, x, reduction='none'
        )
        recon = recon.sum(dim=1)
        kl = ut.kl_normal(m, v, self.z_prior_m, self.z_prior_v)
        nelbo = recon + kl
        nelbo = nelbo.mean()
        recon = recon.mean()
        kl = kl.mean()
        return nelbo, kl, recon
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   niwae, kl, rec
        #
        # HINT: The summation over m may seem to prevent us from
        # splitting the ELBO into the KL and reconstruction terms, but instead consider
        # calculating log_normal w.r.t prior and q
        ################################################################################
        ### START CODE HERE ###
        x_dup = ut.duplicate(x, iw)
        
        m, v = self.enc(x_dup)
        z = ut.sample_gaussian(m, v)
        logits = self.dec(z)
        
        recon_dup = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, x_dup, reduction='none'
        ).sum(dim=-1)
        log_p_x_z = -recon_dup 
        
        log_p_z = ut.log_normal(z, self.z_prior_m, self.z_prior_v)
        log_q_z_x = ut.log_normal(z, m, v)
        
        log_w = log_p_x_z + log_p_z - log_q_z_x
        
        log_w_reshaped = log_w.view(iw, -1)
        
        iwae_bound = ut.log_mean_exp(log_w_reshaped, dim=0)
        niwae = -iwae_bound.mean()
        
        recon_reshaped = recon_dup.view(iw, -1).mean(dim=0)
        kl_dup = log_q_z_x - log_p_z
        kl_reshaped = kl_dup.view(iw, -1).mean(dim=0)
        
        return niwae, kl_reshaped.mean(), recon_reshaped.mean()
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
