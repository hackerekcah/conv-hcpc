import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from utils.utilities import calculate_confusion_matrix, plot_confusion_matrix, calculate_accuracy


class AssVis (object):
    def __init__(self, ckpt_file, model=None):
        self.ckpt_file = ckpt_file
        self.model_state_dict = torch.load(ckpt_file)['model_state_dict']
        self.hist_list = torch.load(ckpt_file)['histories']

        if model:
            self.bind_model(model)
        else:
            # set CNN3BiGRU as default model
            from net_archs import MLP
            from utils.wrapper import ArgWrapper
            default_model = MLP(
                args=ArgWrapper(hidden_size=128)
            )
            self.bind_model(default_model)

        from data_manager.ass_manager18 import TaskbAssManager
        from data_manager.ass_stdrizer18 import TaskbAssStandarizer
        self.data_manager = TaskbAssManager()
        self.data_standarizer = TaskbAssStandarizer(data_manager=self.data_manager)

    def bind_model(self, model):
        """
        bind model, automatically move to cuda
        :param model:
        :return:
        """
        self.model = model
        self.model.load_state_dict(self.model_state_dict, strict=False)
        # explicit move to cuda
        self.model.cuda()
        # set as evaluation mode
        self.model.eval()
        return self

    def phistory(self):
        """
        plot train/val histories
        :return:
        """
        assert self.hist_list is not None
        for hist in self.hist_list:
            hist.plot()

    def play(self, wav_name):
        """
        show an inline audio player below the notebook shell
        :param wav_name:
        :return:
        """
        audio_root = '/data/songhongwei/dcase2018_baseline/task1/datasets/' \
                     'TUT-urban-acoustic-scenes-2018-mobile-development/audio'
        import IPython.display as ipd
        import os
        obj = ipd.Audio(filename=os.path.join(audio_root, wav_name), autoplay=False)

        # need to call display() explicitly if not returned obj to notebook cell
        from IPython.display import display
        display(obj)

    def show(self, wav_name):
        """
        show audio player, ass_vectors, instance predictions and bag predictions
        :param wav_name:
        :return:
        """
        # only support device a
        assert wav_name[-5] == 'a'
        self.play(wav_name=wav_name)
        # (1, Seg, Fea)
        ass_vec = self.data_standarizer.load_normed_ass_by_name(wav_name=wav_name, norm_device='a')
        ass_vec = ass_vec.astype(float)

        self.model.plot(ass_vec, wav_name=wav_name)

    def confusion(self):
        # numpy, x.shape (N, Seg, Fea)
        x, y = self.data_standarizer.load_dev_standrized(mode='test', device='a')

        # numpy predict, (N, C)
        bag_pred = self.model.predict(x, verbose=False, batch_size=128)

        pred_int = bag_pred.argmax(axis=1)
        y_int = y.argmax(axis=1)

        acc = np.mean(pred_int == y_int)

        cf_matrix = calculate_confusion_matrix(y_int, pred_int, classes_num=bag_pred.shape[1])
        labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
                  'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
        class_wise_accuracy = calculate_accuracy(y_int, pred_int, bag_pred.shape[1])

        plot_confusion_matrix(confusion_matrix=cf_matrix,
                              title='Confusion Matrix, Acc{:.3f}'.format(acc),
                              labels=labels,
                              values=class_wise_accuracy)

    def dump_result(self, mode='train', filename=None):
        """
        pickle a dictionary {'fname', 'bag_preds', 'label'} to the given path
        :param mode:
        :param filename:
        :return:
        """
        if not filename:
            filename = self.ckpt_file.replace('.tar', '_' + mode + '_result.pkl')

        if os.path.exists(filename):
            print("result file already exist, no pickle")
            return

        # numpy, x.shape (N,Seg,F)
        x, y, fname = self.data_standarizer.load_dev_fname_standrized(mode=mode, device='a')

        bag_preds = self.model.predict(x, verbose=False, batch_size=128)
        results = dict(fname=fname,
                       fname_encoder=self.data_manager.fname_encoder,
                       bag_preds=bag_preds,
                       label=y)
        import pickle
        pickle.dump(results, open(filename, 'wb'))
        print("result file pickle to", filename)


class ResultAnalyzer (object):
    """
    1. load result file, which contains a dict of {'bag_preds','fname', 'fname_encoder', 'label'}
    2. return a lis of file, given likelihood thresh
    3. plot likelihood cumulative distribution function
    """

    def __init__(self, train_pkl, val_pkl):
        self.train_result = pickle.load(open(train_pkl, 'rb'))
        self.val_result = pickle.load(open(val_pkl, 'rb'))

    def _cdf_value(self, preds, point, ytype='count'):
        """
        get the cdf value of a point in preds
        :param preds: list or np array
        :param point: point prediction value
        :param ytype: 'count' or 'percent'
        :return:
        """
        # NOTE: preds need to be sorted
        from bisect import bisect_left
        # add small value to include the point itself
        if ytype == 'count':
            return bisect_left(preds, point+1e-8)
        elif ytype == 'percent':
            return bisect_left(preds, point+1e-8) / len(preds)

    def _cdf_list(self, preds, ytype='count'):
        """
        return cdf value as a list
        :param preds: list or np array
        :param ytype: 'count' or 'percent'
        :return:
        """
        # sort in ascend order
        preds = sorted(preds)
        return [self._cdf_value(preds, p, ytype) for p in preds]

    def plot_cdf(self, mode='train', ytype='count'):
        """
        plot cumulative density function of the likelihood of true label
        :param mode:
        :param ytype: 'count' or 'percent'
        :return:
        """
        if mode == 'train':
            # (N, C)
            bag_preds = self.train_result['bag_preds']
            true_label = self.train_result['label']
            likelihood = bag_preds[np.where(true_label == 1)]
            likelihood = sorted(likelihood)
            plt.plot(likelihood, self._cdf_list(preds=likelihood, ytype=ytype))
            plt.title("{} cdf of {}".format(ytype, mode))
            plt.show()
        elif mode == 'test':
            # (N, C)
            bag_preds = self.val_result['bag_preds']
            true_label = self.val_result['label']
            likelihood = bag_preds[np.where(true_label == 1)]
            likelihood = sorted(likelihood)
            plt.plot(likelihood, self._cdf_list(preds=likelihood, ytype=ytype))
            plt.title("{} cdf of {}".format(ytype, mode))
            plt.show()

    def files_by_likelihood(self, mode='train', l=0, h=1):
        """
        return file name list, according to likelihood of true label between [l, h]
        :param mode:
        :param l:
        :param h:
        :return:
        """
        if mode == 'train':
            fname = self.train_result['fname']
            fname_encoder = self.train_result['fname_encoder']
            bag_preds = self.train_result['bag_preds']
            true_label = self.train_result['label']
            likelihood = bag_preds[np.where(true_label == 1)]
            return fname_encoder.inverse_transform(fname[np.where((likelihood > l) & (likelihood < h))])
        elif mode == 'test':
            fname = self.val_result['fname']
            fname_encoder = self.val_result['fname_encoder']
            bag_preds = self.val_result['bag_preds']
            true_label = self.val_result['label']
            likelihood = bag_preds[np.where(true_label == 1)]
            return fname_encoder.inverse_transform(fname[np.where((likelihood > l) & (likelihood < h))])

    def confidence_hist(self, mode='train'):
        """
        plot histogram of confidence, and the percentage of true label
        :param mode:
        :return:
        """
        if mode == 'train':
            # (N, C)
            bag_preds = self.train_result['bag_preds']
            true_label = self.train_result['label']
            pred_max = np.max(bag_preds, axis=1)
            plt.hist(x=pred_max, bins=np.arange(0.0, 1.1, step=0.1), edgecolor='black', linewidth=1)
            self.confidence_pos_hist(mode=mode)
        elif mode == 'test':
            # (N, C)
            bag_preds = self.val_result['bag_preds']
            true_label = self.val_result['label']
            pred_max = np.max(bag_preds, axis=1)
            plt.hist(x=pred_max, bins=np.arange(0.0, 1.1, step=0.1), edgecolor='black', linewidth=1)
            self.confidence_pos_hist(mode=mode)

    def confidence_pos_hist(self, mode='train'):
        """
        plot the true prediction confidence histogram
        :param mode:
        :return:
        """
        if mode == 'train':
            # (N, C)
            bag_preds = self.train_result['bag_preds']
            true_label = self.train_result['label']
            pred_int = bag_preds.argmax(axis=1)
            y_int = true_label.argmax(axis=1)

            pred_max = np.max(bag_preds, axis=1)

            pred_max_pos = pred_max[np.where(pred_int == y_int)]
            plt.hist(x=pred_max_pos, bins=np.arange(0.0, 1.1, step=0.1),
                     facecolor='orange', edgecolor='black', linewidth=1)
        elif mode == 'test':
            # (N, C)
            bag_preds = self.val_result['bag_preds']
            true_label = self.val_result['label']
            pred_int = bag_preds.argmax(axis=1)
            y_int = true_label.argmax(axis=1)

            pred_max = np.max(bag_preds, axis=1)

            pred_max_pos = pred_max[np.where(pred_int == y_int)]
            plt.hist(x=pred_max_pos, bins=np.arange(0.0, 1.1, step=0.1),
                     facecolor='orange', edgecolor='black', linewidth=1)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
