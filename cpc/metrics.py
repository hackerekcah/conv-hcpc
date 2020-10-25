import torch
import logging


class CpcMetric:
    """
    Monitor CPC accuracy and losses.
    Record acc for different prediction steps.
    """
    def __init__(self, id, prediction_steps):
        """
        :param id: cpc id
        :param prediction_steps:
        """
        self.cpc_id = id
        self.prediction_steps = prediction_steps
        self.m = dict()
        self.restart()

    def restart(self):
        self.m['correct'] = torch.zeros(self.prediction_steps).cuda()
        self.m['sample_num'] = torch.zeros(1).cuda()

    def accumulate(self, cpc_logit, cpc_label):
        """
        :param cpc_logits: (B*T*K, Ne+1) or (B*F*Vt*K, Ne+1)
        :param cpc_labels: (B*T*K, ) or (B*F*Vt*K, Ne+1)
        :return:
        """
        with torch.no_grad():
            # (B*T*K,)
            pred_cpc = cpc_logit.argmax(-1)
            # (B*T*K,)
            correct_cpc = pred_cpc.eq(cpc_label)
            # (B*T, K)
            correct_cpc = torch.reshape(correct_cpc, (-1, self.prediction_steps))
            # (1,)
            self.m['sample_num'] += correct_cpc.size(0)
            # (K,)
            self.m['correct'] += correct_cpc.sum(dim=0)

    def write(self, writer, epoch, is_train):
        """
        write to tensorboard.
        return mean acc over different prediction steps.
        """
        with torch.no_grad():
            acc = self.m['correct'].div(self.m['sample_num']).cpu()
            for k in range(self.prediction_steps):
                writer.add_scalar("step_acc/cpcid={:d}/pstep={:d}".format(self.cpc_id, k), acc[k].item(), epoch)
            # mean acc over all steps
            mean_acc = self.m['correct'].sum().div(self.m['sample_num'] * self.prediction_steps).cpu().item()
            writer.add_scalar("step_acc/cpcid={:d}/pstep_avg".format(self.cpc_id), mean_acc, epoch)
            # write number of samples
            sample_num = self.m['sample_num'].cpu().item() * self.prediction_steps
            writer.add_scalar("cpc_samples/cpcid={:d}".format(self.cpc_id), sample_num, epoch)
            self.restart()
            if is_train:
                logging.info("Train:step_acc/cpcid={:d}/pstep_avg:{:5.4f}".format(self.cpc_id, mean_acc))
                return {'train_cpc_id_{:d}_acc'.format(self.cpc_id): round(mean_acc, 4)}
            else:
                logging.info("Valid:step_acc/cpcid={:d}/pstep_avg:{:5.4f}".format(self.cpc_id, mean_acc))
                return {'valid_cpc_id_{:d}_acc'.format(self.cpc_id): round(mean_acc, 4)}