import os
import time
import torch
import numpy as np
from tqdm import tqdm
from .evaluator import Metric
from .visualizer import VisualizeLog, VisualizeTSNE
from .builder import build_optimizers, build_schedulers, build_dataloaders

# import sys
# sys.path.append('../')


class Runner(object):
    """training or testing system.

    Args:
        model (nn.Module): the model to train/test.
        logger (logger): the logger for train/test.
        exp_dir (str): work dir to save checkpoint or log file.
        eval_cfg (dict): the config dict of evaluation.
        optim_cfg (dict): the config dict of optimizer.
        sched_cfg (dict): the config dict of learning rate schedule.
        check_cfg (dict): the config dict of checkpoint.
    """
    def __init__(self,
                 model,
                 logger,
                 exp_dir,
                 log_cfg=None,
                 test_cfg=None,
                 eval_cfg=None,
                 optim_cfg=None,
                 sched_cfg=None,
                 check_cfg=None):
        assert isinstance(exp_dir, str)
        self.model = model
        self.logger = logger
        self.log_cfg = log_cfg
        self.test_cfg = test_cfg
        self.eval_cfg = eval_cfg
        self.optim_cfg = optim_cfg
        self.sched_cfg = sched_cfg
        self.check_cfg = check_cfg
        self.exp_dir = os.path.abspath(exp_dir)
        self._total_epoch = 0
        self._total_iter = 0
        self._iter_time = 0
        self._epoch = 0
        self._iter = 0
        self._eta = 0
        self._lr = 0

        os.makedirs(os.path.expanduser(self.exp_dir), exist_ok=True)

        if self.optim_cfg is not None:
            self._warmup = self.sched_cfg.pop('warmup', 0)
            self.optimizer = build_optimizers(self.optim_cfg, self.model)
            self.scheduler = build_schedulers(self.sched_cfg, self.optimizer)
            if self.log_cfg.plog_cfg is not None:
                self.vis_log = VisualizeLog(self.exp_dir, self.log_cfg.plog_cfg)

        self._score = np.zeros((self.check_cfg.pop('save_topk', 1),), dtype=np.float32)
        self._init_model(self.check_cfg.resume_from, self.check_cfg.load_from)
        self.metric = Metric(logger, self.exp_dir, eval_cfg)

        if self.eval_cfg.tsne_cfg is not None:
            self.vis_tsne = VisualizeTSNE(self.exp_dir, self.eval_cfg.tsne_cfg)

        self.dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def _lr_step(self):
        """update learning rate"""
        if self._warmup > self._iter:
            init_lrs = [v['initial_lr'] for v in self.optimizer.param_groups]
            for param_group, lr in zip(self.optimizer.param_groups, init_lrs):
                param_group['lr'] = lr / self._warmup * self._iter
        self._lr = self.optimizer.param_groups[0]['lr']

    def _log_infos(self, output):
        """log training / validation info"""
        eta = (self._total_iter - self._iter) * self._iter_time / self.log_cfg.interval
        mins = '{:2d}'.format(int((eta % 3600) / 60))
        hours = '{:2d}'.format(int(eta / 3600))
        lr = '{:6f}'.format(self._lr)
        self._iter_time = 0
        info = f'Epoch: {self._epoch}, Iter: {self._iter}, ETA: {hours}h{mins}min, Lr: {lr},'
        for k, v in output.items():
            if k == 'loss' or k == 'val_loss':
                continue
            loss = '{:.5f}'.format(v.mean().detach().item())
            info += f' {k}: {loss},'
        try:
            info += ' val_loss: {:.5f}'.format(output['val_loss'].mean().detach().item())
        except:
            info += ' loss: {:.5f}'.format(output['loss'].mean().detach().item())
        self.logger.info(info)

    def _init_model(self, resume_from=None, load_from=None):
        """initialize model"""
        if resume_from is not None:
            checkpoint = torch.load(resume_from, map_location='cpu')
            self._iter = checkpoint['iter']
            self._epoch = checkpoint['epoch']
            self._score = checkpoint['score']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(f'Resume from {resume_from}, {self._epoch} epoch, {self._iter} iter')
        elif load_from is not None:
            checkpoint = torch.load(load_from, map_location='cpu')
            self._iter = checkpoint['iter']
            self._epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(f'Load from {load_from}, {self._epoch} epoch, {self._iter} iter')

    def _save_model(self, score=None, filename=None):
        """save model"""
        if score is not None:
            if score < self._score[-1]:
                return
            for k in range(len(self._score)):
                if score >= self._score[k]:
                    filename = os.path.join(self.exp_dir, f'top{k + 1}_model.pth')
                    if os.path.exists(filename) and k < len(self._score) - 1:
                        self._score[k + 1] = self._score[k]
                        filename_next = os.path.join(self.exp_dir, f'top{k + 2}_model.pth')
                        os.system(f'mv {filename} {filename_next}')
                    self._score[k] = score
                    break
        elif filename is None:
            filename = os.path.join(self.exp_dir, f'epoch{self._epoch}_iter{self._iter}.pth')
        else:
            filename = os.path.join(self.exp_dir, filename)
        checkpoint = dict(
            iter=self._iter,
            epoch=self._epoch,
            score=self._score,
            state_dict=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict())
        self.logger.info(f'The temporal saving model is: {filename}')
        torch.save(checkpoint, filename)

    def _get_total_iter(self, dataloader, step_cfg=None):
        """get total iter"""
        if step_cfg is None:
            return self._total_epoch * len(dataloader)
        else:
            ir, hr, dr = step_cfg.init_rate, step_cfg.hem_rate, step_cfg.decay_rate
            train_len = int(ir * len(dataloader))
            test_len = len(dataloader) - train_len
            total_iter = 0
            if self._total_epoch < step_cfg.interval:
                self._total_iter = self._total_epoch * len(dataloader)
            for i in range(int(self._total_epoch / step_cfg.interval)):
                total_iter += train_len * step_cfg.interval
                train_len += test_len * hr
                test_len *= (1 - hr)
                hr *= dr
        self._total_iter = int(total_iter)
        return int(total_iter)

    def _train_epoch(self, dataloader):
        """train one epoch"""
        start_time = time.time()
        for i, data in enumerate(dataloader):
            self._iter += 1

            output = self.model(**data)
            self.optimizer.zero_grad()
            output['loss'].mean().backward()

            self.optimizer.step()
            self._lr_step()

            self._iter_time += time.time() - start_time
            start_time = time.time()

            if self._iter % self.log_cfg.interval == 0:
                self._log_infos(output)

            if self._iter % self.eval_cfg.interval == 0:
                score, val_loss = self.val()
                self._log_infos(val_loss)
                self._save_model(score)
                self._save_model(filename='latest.pth')

            if self._iter % self.check_cfg.interval == 0:
                self._save_model()

    @torch.no_grad()
    def val(self):
        """val method"""
        preds = list()
        labels = list()
        val_loss = dict(val_loss=0.0)
        self.model.eval()
        for data in tqdm(self.val_dataloader):
            output = self.model(**data) # [logits, labels, losses, feats (optional)]
            preds.append(output[0].argmax(dim=-1).detach().cpu().numpy())
            labels.append(output[1].detach().cpu().numpy()[:, 0])
            val_loss['val_loss'] = (val_loss['val_loss'] + output[2]['loss'].mean())/2.0
        self.model.train()
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        score, eval_dict = self.metric(preds, labels)
        return score, val_loss

    @torch.no_grad()
    def test(self, dataloader, filename=None, log_info=True):
        """test method"""
        feats = list()
        preds = list()
        labels = list()
        self.model.eval()
        for data in tqdm(dataloader):
            output = self.model(data['img'], data['label']) # [logits, labels, losses, feats (optional)]
            preds.append(output[0].argmax(dim=-1).detach().cpu().numpy())
            labels.append(output[1].detach().cpu().numpy()[:, 0])
            if len(output) > 3:
                feats.append(output[-1].detach().cpu().numpy())
        self.model.train()
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        if len(feats) > 0:
            feats = np.concatenate(feats)
            if self.eval_cfg.tsne_cfg is not None:
                self.vis_tsne(feats, labels)
        self.metric(preds, labels, filename=filename, log_info=log_info)
        return preds, labels

    def train(self, dataloader, cfg):
        """train method"""
        self._total_epoch = cfg.total_epochs
        self._total_iter = self._get_total_iter(dataloader)
        self.logger.info(f'Start training from the {self._epoch} Epoch, {self._iter} Iter.')
        self.logger.info(f'Total {self._total_epoch} Epochs, {self._total_iter} Iters, {len(dataloader)} Iter/Epoch.')

        self.model.train()
        start_epoch = self._epoch
        for epoch in range(start_epoch, self._total_epoch):
            self.vis_log(self.log_cfg.filename)
            
            self._train_epoch(dataloader)

            if self.log_cfg.plog_cfg is not None:
                self.vis_log(self.log_cfg.filename)

            self._epoch += 1
            self.scheduler.step()
            self._save_model(filename='latest.pth')

        self.logger.info('End of training!')

    def train_step(self, dataloader, cfg):
        """train step method"""
        self._total_epoch = cfg.total_epochs
        self._total_iter = self._get_total_iter(dataloader, cfg.step_cfg)
        self.logger.info(f'Start training from the {self._epoch} Epoch, {self._iter} Iter.')
        self.logger.info(f'Total {self._total_epoch} Epochs, {self._total_iter} Iters, {len(dataloader)} Iter/Epoch.')

        train_dataset = dataloader.dataset
        test_dataset = self.test_dataloader.dataset
        data_infos = np.array(train_dataset.data_infos)
        pinfos, ninfos = data_infos[train_dataset.flag == 0], data_infos[train_dataset.flag != 0]
        np.random.shuffle(pinfos)
        np.random.shuffle(ninfos)

        ir, hr, dr = cfg.step_cfg.init_rate, cfg.step_cfg.hem_rate, cfg.step_cfg.decay_rate
        train_dataset.data_infos = pinfos[:int(ir * len(pinfos))].tolist() + ninfos[:int(ir * len(ninfos))].tolist()
        pinfos, ninfos = pinfos[int(ir * len(pinfos)):], ninfos[int(ir * len(ninfos)):]

        self.model.train()
        start_epoch = self._epoch
        self.logger.info(f'Train sub dataset: {train_dataset.set_group_flag()}')
        dataloader = build_dataloaders(cfg.data.train_loader, train_dataset)
        for epoch in range(start_epoch, self._total_epoch):
            
            self._train_epoch(dataloader)

            self._epoch += 1
            if self._epoch % cfg.step_cfg.interval == 0:

                test_dataset.data_infos = pinfos.tolist() + ninfos.tolist()
                self.logger.info(f'Test sub dataset: {test_dataset.set_group_flag()}')
                test_dataloader = build_dataloaders(cfg.data.test_loader, test_dataset)

                preds, labels = self.test(test_dataloader, log_info=False)

                pinds = np.argsort(preds[:len(pinfos)])
                ninds = np.argsort(-preds[len(pinfos):])

                phn, nhn = int(hr * len(pinfos)), int(hr * len(ninfos))
                phs, nhs = preds[pinds[: phn]].mean(), preds[ninds[: nhn] + len(pinfos)].mean()

                hard_infos = pinfos[pinds[: phn]].tolist() + ninfos[ninds[: nhn]].tolist()
                train_dataset.data_infos.extend(hard_infos)
                self.logger.info(f'Train sub dataset: {train_dataset.set_group_flag()}')
                dataloader = build_dataloaders(cfg.data.train_loader, train_dataset)

                pinfos, ninfos = pinfos[pinds[phn:]], ninfos[ninds[nhn:]]
                hr *= dr

                self.logger.info(f'Phard num: {phn}, score: {phs}, Nhard num: {nhn}, score: {nhs}')

            if self.log_cfg.plog_cfg is not None:
                self.vis_log(self.log_cfg.filename)

            self.scheduler.step()
            self._save_model(filename='latest.pth')

        self.logger.info('End of step training!')