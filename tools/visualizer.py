import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold


class VisualizeTSNE(object):
    """Visualize TSNE for features.

    Args:
        work_dir (str): work dir to log file or result file.
        tsne_cfg (dict): the config dict of tsne.
        figsize (tuple): figure size to draw.
    """
    def __init__(self,
                 work_dir,
                 tsne_cfg=None,
                 figsize=(10, 10)):
        self.eval_cfg = tsne_cfg
        self.work_dir = os.path.join(work_dir, 'results')
        os.makedirs(self.work_dir, exist_ok=True)

        self.marks = tsne_cfg.pop('marks', None)
        self.filename = tsne_cfg.pop('filename', 'tsne.png')
        self.tsne = manifold.TSNE(init='pca', n_components=2, perplexity=15, random_state=501)
        self.figure = plt.figure(figsize=figsize)
        self.colors = ['r', 'g', 'b', 'y', 'c', 'm']

    def __call__(self, feats, labels):
        """call function.
        Args:
            feats (np.array()): the features shape of (N, d).
            labels (np.array()): the labels shape of (N, ).

        """
        groups = np.bincount(labels.astype(np.uint8))
        if self.marks is not None:
            assert len(groups) == len(self.marks)
        positions = self.tsne.fit_transform(feats)
        positions = (positions - positions.min(0)) / (positions.max(0) - positions.min(0))

        ax = self.figure.add_subplot(111)
        for i, g in enumerate(groups):
            if g == 0:
                continue
            posi = positions[labels == i, :]
            if self.marks is None:
                ax.scatter(posi[:, 0], posi[:, 1], s=20, c=self.colors[i % 6], marker='o', alpha=0.8)
            else:
                ax.scatter(posi[:, 0], posi[:, 1], **self.marks[i])
        ax.legend(prop=dict(size=14))
        plt.tick_params(labelsize=15)
        plt.savefig(os.path.join(self.work_dir, self.filename))


class VisualizeLog(object):
    """Visualize train logs.

    Args:
        work_dir (str): work dir to log file or result file.
        figsize (tuple): figure size to draw.
        eval_types (str, list): eval types to plot.
        loss_types (list, optional): loss types to plot.

    """
    def __init__(self,
                 work_dir,
                 plog_cfg=None,
                 figsize=(15, 10)):
        self.figsize = figsize
        self.plog_cfg = plog_cfg.copy()
        self.loss_types = self.plog_cfg .pop('loss_types', None)
        self.eval_types = self.plog_cfg .pop('eval_types', 'trr@1e-2')
        if not isinstance(self.eval_types, list):
            self.eval_types = [self.eval_types]

        self.work_dir = os.path.join(work_dir, 'results')
        os.makedirs(self.work_dir, exist_ok=True)

        self.colors = ['r', 'g', 'b', 'y', 'c', 'm']
        self.eval_names = ['acc', 'auc', 'acer', 'apcer', 'bpcer',
                           'trr@1e-4', 'trr@1e-3', 'trr@1e-2', 'trr@1e-1']

    def _parse_log(self, log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        iter = 0
        evals = list()
        losses = list()
        loss_names = []
        for line in lines:
            if 'INFO' in line and 'loss:' in line:
                infos = re.split(': |, ', re.split('min, ', line)[1])

                if iter == 0:
                    iter = int(re.findall(r"(?<=Iter: )\d+", line)[0])
                    loss_names = infos[0::2]

                loss = list(map(float, infos[1::2]))
                loss.append(iter)
                losses.append(loss)
                iter += 20

            elif '* >>' in line and 'acc' not in line:
                res = list(map(float, re.findall(r"\d+\.?\d*", line)))
                res.append(iter)
                evals.append(res)

        return evals, losses, loss_names

    def __call__(self, log_file):
        """call function

        Args:
            log_file (str): log file dir.
        """
        name = os.path.basename(log_file).split('.')[0]

        evals, losses, loss_names = self._parse_log(log_file)

        evals = np.array(evals)
        losses = np.array(losses)
        if len(evals) == 0 or evals.shape[1] != 10:
            return 

        plt.figure(num=0, figsize=self.figsize)
        plt.title('Evaluation')
        plt.switch_backend('agg')
        for i, k in enumerate(self.eval_names):
            if k not in self.eval_types:
                continue
            plt.plot(evals[:, -1], evals[:, i], self.colors[i % 6], label=k)
        plt.grid(ls='--')
        plt.legend(loc='lower right')
        plt.ylabel('Eval Score')
        plt.xlabel('Iters')
        plt.savefig(os.path.join(self.work_dir, f'{name}_eval.png'))

        if self.loss_types is None:
            return

        plt.figure(num=1, figsize=self.figsize)
        plt.title('Loss')
        plt.switch_backend('agg')
        for i, k in enumerate(loss_names):
            if k in self.loss_types:
                plt.plot(losses[:, -1], losses[:, i], self.colors[i % 6], label=k)
        plt.grid(ls='--')
        plt.legend(loc='upper right')
        plt.ylabel('Loss')
        plt.xlabel('Iters')
        plt.savefig(os.path.join(self.work_dir, f'{name}_loss.png'))
