# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="/content/drive/MyDrive/maskscoring_rcnn_origin/models",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            print("找不到checkpoint，从头加载模型")
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        """
          更新模型的参数
        """
        #orgin_dict = self.model.state_dict()
        #resnet_preloadingdict = torch.load("/content/drive/MyDrive/mask scoring rcnn/res50-model_000100.pth")
        #resnet90000_dict =  {k: v for k, v in resnet90000_dict.items() if k in orgin_dict}
        #orgin_dict.update(resnet90000_dict)
        #self.model.load_state_dict(orgin_dict)
        #checkpoint = self._load_file(f)
        #self._load_model(checkpoint,f)
        #pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #model_dict.update(pretrained_dict)
        #self.resnet50.load_state_dict(model_dict)
        
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        
        """
          使用default指定的optimizer和scheduler，不加载checkpoing中的optimizer和scheduler
        """
        # if "optimizer" in checkpoint and self.optimizer:
        #    self.logger.info("Loading optimizer from {}".format(f))
        #    self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        # if "scheduler" in checkpoint and self.scheduler:
        #    self.logger.info("Loading scheduler from {}".format(f))
        #    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        #print("从{}更新checkpoint".format(f))
        #self.logger.info("Updating checkpoint from {}".format(f))
        #msrcnn_model_dict = self.model.state_dict()
        #pretrained_dict =  {k: v for k, v in checkpoint.items() if k in msrcnn_model_dict}
        #msrcnn_model_dict.update(pretrained_dict)
        #self.model.load_state_dict(msrcnn_model_dict)
        #load_state_dict(self.model, msrcnn_model_dict)
        print("---------------------------------")
        copy_checkpoint = checkpoint
        print(copy_checkpoint["model"].keys())
        # del copy_checkpoint["model"]["module.roi_heads.box.predictor.cls_score.weight"]
        # del copy_checkpoint["model"]["module.roi_heads.box.predictor.cls_score.bias"]
        # del copy_checkpoint["model"]["module.roi_heads.box.predictor.bbox_pred.weight"]
        # del copy_checkpoint["model"]["module.roi_heads.box.predictor.bbox_pred.bias"]
        orgin_dict = self.model.state_dict()
        copy_checkpoint =  {k: v for k, v in copy_checkpoint.items() if k in orgin_dict}
        orgin_dict.update(copy_checkpoint)
        print("---------------------------------")
        #self.model.load_state_dict(orgin_dict)
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        print("加载文件_load_file{}".format(f))
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f, model_dir=self.cfg.MODEL.PRETRAINED_MODELS)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
