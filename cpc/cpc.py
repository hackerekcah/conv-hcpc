import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
logging.basicConfig(
    # format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class CPC(nn.Module):
    """
    The Contrastive Predictive Coding loss.
    """
    def __init__(self, d_z, d_c, prediction_steps, n_negatives, lrf, causal_offset, hop,
                 softmax_tau, is_conditional,
                 cpc_dropout,
                 cpc_hidden_units,
                 cpc_target_dim,
                 cpc_id):
        """
        :param d_z: dimension of latent vector
        :param d_c: dimension of context vector
        :param prediction_steps:
        :param n_negatives:
        :param lrf: left receptive field in "number of c", skip initial context bcz lack of information.
        :param causal_offset: offset needed to make c causal aligned with z.
        :param hop: hop in "number of z" for to the next c
        :param cpc_dropout: context dropping.
        :param cpc_target_dim: mapping z and c into a common space.
        :param softmax_tau: temperature
        :param is_conditional: concat z_t and c_t to predict z_{t+k}
        :param cpc_id: unique id for cpc module
        """
        super(CPC, self).__init__()
        self.id = cpc_id
        logger.info("cpc_id: {:d}, causal offset: {:d}".format(self.id, causal_offset))
        logger.info("cpc_id: {:d}, hop: {:d}".format(self.id, hop))
        logger.info("cpc_id: {:d}, dropout: {:3.1f}".format(self.id, cpc_dropout))
        logger.info("cpc_id: {:d}, cpc_hidden_units: {:d}".format(self.id, cpc_hidden_units))

        if cpc_target_dim == 'None':
            self.cpc_target_dim = None
        else:
            self.cpc_target_dim = cpc_target_dim

        self._log_once = False

        self.is_conditional = is_conditional

        if self.cpc_target_dim is not None and self.cpc_target_dim != 'None':
            self.mapz = nn.Conv1d(in_channels=d_z, out_channels=self.cpc_target_dim,
                                  kernel_size=1, stride=1)
            d_z = self.cpc_target_dim

        if self.is_conditional:
            d_in = d_c + d_z
        else:
            d_in = d_c

        if cpc_hidden_units > 0:
            self.proj_heads = nn.Sequential(
                nn.Dropout(p=cpc_dropout),
                nn.ConvTranspose2d(
                    d_in, cpc_hidden_units, (1, prediction_steps)
                ),
                nn.GroupNorm(num_groups=1, num_channels=cpc_hidden_units),
                nn.ReLU(),
                nn.Conv2d(cpc_hidden_units, d_z, kernel_size=1, bias=False)
            )
        else:
            # using transposed conv to make K predictions at the same time.
            self.proj_heads = nn.Sequential(
                nn.Dropout(p=cpc_dropout),
                # (B, d_c+dz, T_c, 1) -> (B, d_out, T_c, K)
                nn.ConvTranspose2d(
                    d_in, d_z, (1, prediction_steps)
                )
            )

        self.causal_offset = causal_offset
        self.lrf = lrf
        self.hop = hop
        self.prediction_steps = prediction_steps
        self.n_negatives = n_negatives
        self.softmax_tau = softmax_tau

    def forward(self, c, z):
        """
        :param c: (B, d_c, Tc)
        :param z: (B, d_z, Tz)
        :return:
        """
        c = c[..., self.lrf:]
        if self.cpc_target_dim is not None:
            z = self.mapz(z)

        # (B, vt, K, 1+Ne)
        pos_neg_idx = self._get_sample_idx(z)

        if not self._log_once:
            logger.info("Num of contexts is {:d}".format(pos_neg_idx.size(1)))
            self._log_once = not self._log_once

        if self.is_conditional:
            pred = self._get_condition_pred(c, z, pred_num=pos_neg_idx.size(1))
        else:
            # (B, dz, T, K, 1+Ne)
            pred = self._get_pred(c, pred_num=pos_neg_idx.size(1))

        # (B, dz, T, K, 1+Ne)
        samples = self._get_samples_from_idx(pos_neg_idx=pos_neg_idx, z=z)

        # (B, T, K, 1+Ne)
        s = F.cosine_similarity(pred, samples, dim=1)
        # (B*T*K, 1+Ne)
        cpc_logits = torch.reshape(s, (-1, s.size(-1)))
        # scale cpc_logits by give tau
        cpc_logits = torch.div(cpc_logits, self.softmax_tau)

        # (B*Tc-1*K,), labels are zeros, bcz true future is always the first in 1+N samples.
        cpc_label = torch.zeros(size=(cpc_logits.size(0),), dtype=torch.int64).to('cuda')

        return cpc_logits, cpc_label

    def _get_condition_z(self, z, pred_num):
        bsz, d_z, t_z = z.size()
        with torch.no_grad():

            condition_idx = self._moving_window(torch.arange(t_z),
                                                length=1,
                                                hop=self.hop,
                                                start=self.lrf * self.hop)
            # (vt,)
            condition_idx = condition_idx[:pred_num, :].squeeze(dim=-1)
            # (B, dz, vt)
            condition_idx = condition_idx.view(1, 1, -1).expand([bsz, d_z, -1])

            # (B, dz, vt)
            condition_z = torch.gather(z, dim=-1,
                                       index=condition_idx.to('cuda'))
                # (B, dz, vt, 1)
        return condition_z.unsqueeze(-1)

    def _get_condition_pred(self, c, z, pred_num):

        # (B, d_c, Tc) -> (B, d_c, Tc, 1)
        c = c.unsqueeze(-1)
        # (B, dc, T, 1)
        c = c[..., :pred_num, :]
        # (B, dz, T, 1)
        condition_z = self._get_condition_z(z, pred_num)
        # (B, dz+dc, T, 1)
        cz = torch.cat([c, condition_z], dim=1)

        # (B, d_c+dz, T, 1) -> (B, d_z, T, K)
        pred = self.proj_heads(cz)

        # (B, d_z, T, K, Ne+1)
        pred = pred.unsqueeze(-1).expand((-1, -1, -1, -1, self.n_negatives + 1))
        return pred

    def _get_pred(self, c, pred_num):
        # (B, d_c, Tc) -> (B, d_c, Tc, 1)
        c = c.unsqueeze(-1)
        # (B, dc, T, 1)
        c = c[..., :pred_num, :]
        # (B, d_c, T, 1) -> (B, d_z, T, K)
        pred = self.proj_heads(c)

        # (B, d_z, T, K, Ne+1)
        pred = pred.unsqueeze(-1).expand((-1, -1, -1, -1, self.n_negatives + 1))
        return pred

    def _moving_window(self, x, length, hop, start):
        pidx = [x[i: i + length] for i in range(start, (len(x) + 1) - length, hop)]
        return torch.stack(pidx, dim=0)

    def _get_sample_idx(self, z):
        bsz, d_z, t_z = z.size()

        with torch.no_grad():
            # (vt, K)
            p_idx = self._moving_window(torch.arange(t_z),
                                        length=self.prediction_steps,
                                        hop=self.hop,
                                        start=self.lrf * self.hop + self.causal_offset)
            # (B, vt, K, 1)
            p_idx = p_idx.unsqueeze(0).expand([bsz, -1, -1]).unsqueeze(-1)
            # (B, vt, K, Ne)
            n_size = p_idx.size()[:3] + (self.n_negatives,)
            n_idx = torch.randint(low=0, high=t_z, size=n_size)
            count = 0
            while (n_idx == p_idx).any():
                # condition=true, get n_idx, else regenerate.
                n_idx = torch.where(n_idx != p_idx, n_idx, torch.randint(low=0, high=t_z, size=n_size))
                count += 1
                if count > 10:
                    logging.warn("Collisions over 10 times")
                    break

            # (B, vt, K, 1+Ne)
            pos_neg_idx = torch.cat([p_idx, n_idx], dim=-1)
            return pos_neg_idx

    def _get_samples_from_idx(self, pos_neg_idx, z):
        """
        :param pos_neg_idx: (B, vt, K, 1+Ne)
        :param z: (B, dz, T)
        :return:
        """
        bsz, vt, _, pos_neg_num = pos_neg_idx.size()
        with torch.no_grad():
            d_z = z.size(1)

            # (B, T'=vt*K*(1+Ne))
            pos_neg_idx = torch.flatten(pos_neg_idx, start_dim=1).contiguous()
            # (B, dz, T'=vt*K*(1+Ne))
            pos_neg_idx = pos_neg_idx.unsqueeze(1).expand([-1, d_z, -1])
        # (B, dz, T'=vt*K*(1+Ne))
        samples = torch.gather(z, dim=-1, index=pos_neg_idx.to('cuda'))
        # (B, dz, vt, K, 1+Ne)
        samples = samples.view(bsz, d_z, vt, self.prediction_steps, pos_neg_num)
        assert samples.is_contiguous()
        return samples.contiguous()


if __name__ == '__main__':
    bsz = 2
    dsz = 4
    tsz = 6
    z = torch.arange(48).float().view((bsz, dsz, tsz)).to('cuda')
    c = torch.arange(48).float().view((bsz, dsz, tsz)).to('cuda')
    cpc = CPC(d_z=dsz,
              d_c=dsz,
              prediction_steps=3,
              n_negatives=10,
              lrf=0,
              causal_offset=1,
              hop=1,
              softmax_tau=0.1,
              is_conditional=True,
              cpc_dropout=0.1,
              cpc_hidden_units=0,
              cpc_id=0,
              cpc_target_dim=64).to('cuda')
    logit, label = cpc(c, z)
