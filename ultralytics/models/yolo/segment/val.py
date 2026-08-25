# Ultralytics YOLO 🚀, AGPL-3.0 license

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.torch_utils import get_flops, get_num_params


class SegmentationValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml')
        validator = SegmentationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.crack_mious = []
        self.crack_cldices = []

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        """Initialize metrics and select mask processing function based on save_json flag."""
        super().init_metrics(model)
        self.plot_masks = []
        self._model = model  # save model reference for FLOPs/params calculation
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.crack_mious = []
        self.crack_cldices = []

    @staticmethod
    def _soft_skeleton(mask, iterations):
        """Morphological soft skeleton used for topology-aware clDice evaluation."""
        def erode(x):
            return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

        def opened(x):
            return F.max_pool2d(erode(x), kernel_size=3, stride=1, padding=1)

        skeleton = F.relu(mask - opened(mask))
        for _ in range(max(int(iterations) - 1, 0)):
            mask = erode(mask)
            delta = F.relu(mask - opened(mask))
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return skeleton.clamp_(0, 1)

    def _update_crack_structure_metrics(self, pred_masks, pred_conf, gt_masks, shape):
        """Accumulate foreground-union mIoU and clDice for one validation image."""
        if gt_masks.numel():
            target = (gt_masks > 0).any(dim=0, keepdim=True).float()
        else:
            height, width = int(shape[0]), int(shape[1])
            target = torch.zeros((1, height, width), device=self.device)

        if pred_masks is not None and pred_masks.numel():
            keep = pred_conf >= float(self.args.crack_metric_conf)
            prediction = pred_masks[keep].float().amax(dim=0, keepdim=True) if keep.any() else torch.zeros_like(target)
        else:
            prediction = torch.zeros_like(target)

        if target.shape[-2:] != prediction.shape[-2:]:
            target = F.interpolate(target[None], prediction.shape[-2:], mode="nearest")[0]
        target = target.gt(0.5).float()
        prediction = prediction.gt(0.5).float()
        intersection = (target * prediction).sum()
        union = target.sum() + prediction.sum() - intersection
        both_empty = union == 0
        miou = torch.ones_like(union) if both_empty else intersection / union.clamp_min(1.0)

        if target.sum() == 0 and prediction.sum() == 0:
            cldice = target.new_tensor(1.0)
        elif target.sum() == 0 or prediction.sum() == 0:
            cldice = target.new_tensor(0.0)
        else:
            target_4d, prediction_4d = target[None], prediction[None]
            skel_target = self._soft_skeleton(target_4d, self.args.cldice_iters)
            skel_prediction = self._soft_skeleton(prediction_4d, self.args.cldice_iters)
            topology_precision = (skel_prediction * target_4d).sum() / skel_prediction.sum().clamp_min(1.0)
            topology_sensitivity = (skel_target * prediction_4d).sum() / skel_target.sum().clamp_min(1.0)
            cldice = 2 * topology_precision * topology_sensitivity / (
                topology_precision + topology_sensitivity
            ).clamp_min(1e-6)
        self.crack_mious.append(float(miou))
        self.crack_cldices.append(float(cldice))

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 12) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95",
            "mAP75)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95",
            "mAP75)",
        )

    def postprocess(self, preds):
        """Post-processes YOLO predictions and returns output detections with proto."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        return p, proto

    def _prepare_batch(self, si, batch):
        """Prepares a batch for training or inference by processing images and targets."""
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto):
        """Prepares a batch for training or inference by processing images and targets."""
        predn = super()._prepare_pred(pred, pbatch)
        pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=pbatch["imgsz"])
        return predn, pred_masks

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            gt_masks = pbatch.pop("masks")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                self._update_crack_structure_metrics(None, stat["conf"], gt_masks, pbatch["imgsz"])
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Masks
            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            self._update_crack_structure_metrics(pred_masks, predn[:, 4], gt_masks, pbatch["imgsz"])
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_m"] = self._process_batch(
                    predn, bbox, cls, pred_masks, gt_masks, self.args.overlap_mask, masks=True
                )
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

            # Save
            if self.args.save_json:
                pred_masks = ops.scale_image(
                    pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                    pbatch["ori_shape"],
                    ratio_pad=batch["ratio_pad"][si],
                )
                self.pred_to_json(predn, batch["im_file"][si], pred_masks)
            # if self.args.save_txt:
            #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

    def get_stats(self):
        """Return instance AP plus foreground-union crack structure metrics."""
        stats = super().get_stats()
        miou = float(np.mean(self.crack_mious)) if self.crack_mious else 0.0
        cldice = float(np.mean(self.crack_cldices)) if self.crack_cldices else 0.0
        stats["metrics/mIoU(M)"] = miou
        stats["metrics/clDice(M)"] = cldice
        stats["metrics/mask_fitness"] = 0.1 * self.metrics.seg.map50 + 0.9 * self.metrics.seg.map
        return stats

    def finalize_metrics(self, *args, **kwargs):
        """Sets speed and confusion matrix for evaluation metrics."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Return correct prediction matrix.

        Args:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2

        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        if masks:
            if overlap:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:  # boxes
            iou = box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with masks and bounding boxes."""
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=15),  # not set to self.args.max_det due to slow plotting speed
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
        self.plot_masks.clear()

    def pred_to_json(self, predn, filename, pred_masks):
        """
        Save one JSON result.

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                    "segmentation": rles[i],
                }
            )

    def print_results(self):
        """Prints training/validation set metrics per class for instance segmentation with detailed output."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(
            pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())
        )
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels"
            )
            return

        # --- Per-class results for Box ---
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            LOGGER.info("\n--- Per-class Box Results ---")
            box_keys = self.metrics.keys[:5]  # P(B), R(B), mAP50(B), mAP50-95(B), mAP75(B)
            pf_box = "%22s" + "%11i" * 2 + "%11.3g" * 5
            LOGGER.info(pf_box % ("Class", *(["Images", "Instances"] + box_keys)))
            for i, c in enumerate(self.metrics.ap_class_index):
                box_results = self.metrics.box.class_result(i)
                bap75 = self.metrics.box.all_ap[i, 5] if len(self.metrics.box.all_ap) else 0.0
                LOGGER.info(
                    pf_box
                    % (
                        self.names[c],
                        self.nt_per_image[c],
                        self.nt_per_class[c],
                        box_results[0],  # P
                        box_results[1],  # R
                        box_results[2],  # mAP50
                        box_results[3],  # mAP50-95
                        bap75,           # mAP75
                    )
                )

            # --- Per-class results for Mask ---
            LOGGER.info("\n--- Per-class Mask Results ---")
            mask_keys = self.metrics.keys[5:]  # P(M), R(M), mAP50(M), mAP50-95(M), mAP75(M)
            pf_mask = "%22s" + "%11i" * 2 + "%11.3g" * 5
            LOGGER.info(pf_mask % ("Class", *(["Images", "Instances"] + mask_keys)))
            for i, c in enumerate(self.metrics.ap_class_index):
                mask_results = self.metrics.seg.class_result(i)
                sap75 = self.metrics.seg.all_ap[i, 5] if len(self.metrics.seg.all_ap) else 0.0
                LOGGER.info(
                    pf_mask
                    % (
                        self.names[c],
                        self.nt_per_image[c],
                        self.nt_per_class[c],
                        mask_results[0],  # P
                        mask_results[1],  # R
                        mask_results[2],  # mAP50
                        mask_results[3],  # mAP50-95
                        sap75,            # mAP75
                    )
                )

        # --- Structured Instance Segmentation Summary ---
        self._print_seg_summary()

        # --- Confusion matrix plots ---
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    names=self.names.values(),
                    normalize=normalize,
                    on_plot=self.on_plot,
                )

        # --- Save detailed report to file ---
        self._save_detailed_report()

    def _print_seg_summary(self):
        """Print a structured instance segmentation evaluation summary."""
        box = self.metrics.box
        seg = self.metrics.seg
        sep = "=" * 62

        LOGGER.info(f"\n{sep}")
        LOGGER.info("Instance Segmentation Evaluation Summary")
        LOGGER.info(sep)

        # Box metrics
        LOGGER.info("Bounding Box Metrics (Box):")
        LOGGER.info(f"  mAP@0.5:      {box.map50:.4f}")
        LOGGER.info(f"  mAP@0.75:     {box.map75:.4f}")
        LOGGER.info(f"  mAP@0.5:0.95: {box.map:.4f}")
        LOGGER.info(f"  Precision:    {box.mp:.4f}")
        LOGGER.info(f"  Recall:       {box.mr:.4f}")

        LOGGER.info("")

        # Mask metrics
        LOGGER.info("Mask Segmentation Metrics (Mask):")
        LOGGER.info(f"  mAP@0.5:      {seg.map50:.4f}")
        LOGGER.info(f"  mAP@0.75:     {seg.map75:.4f}")
        LOGGER.info(f"  mAP@0.5:0.95: {seg.map:.4f}")
        LOGGER.info(f"  Precision:    {seg.mp:.4f}")
        LOGGER.info(f"  Recall:       {seg.mr:.4f}")
        LOGGER.info(f"  mIoU (union): {np.mean(self.crack_mious) if self.crack_mious else 0.0:.4f}")
        LOGGER.info(f"  clDice:       {np.mean(self.crack_cldices) if self.crack_cldices else 0.0:.4f}")

        LOGGER.info(f"\n  Fitness:      {self.metrics.fitness:.4f}")

        # FPS
        inference_ms = self.speed.get("inference", 0) if hasattr(self, "speed") and self.speed else 0
        if inference_ms > 0:
            LOGGER.info(f"  FPS:          {1000.0 / inference_ms:.1f} (inference only)")

        # Speed
        if hasattr(self, "speed") and self.speed:
            LOGGER.info(
                f"  Speed: {self.speed.get('preprocess', 0):.1f}ms preprocess, "
                f"{self.speed.get('inference', 0):.1f}ms inference, "
                f"{self.speed.get('loss', 0):.1f}ms loss, "
                f"{self.speed.get('postprocess', 0):.1f}ms postprocess per image"
            )
        LOGGER.info(sep)

    def _save_detailed_report(self):
        """Save a detailed instance segmentation evaluation report to a text file."""
        report_path = self.save_dir / getattr(self, "report_filename", "segmentation_eval_report.txt")
        box = self.metrics.box
        seg = self.metrics.seg
        sep = "=" * 62

        # --- Compute model efficiency metrics ---
        model = getattr(self, "_model", None)
        if model is not None:
            try:
                n_params = get_num_params(model)
            except Exception:
                n_params = -1
            try:
                gflops = get_flops(model, imgsz=self.args.imgsz)
            except Exception:
                gflops = -1.0
        else:
            n_params = -1
            gflops = -1.0

        # FPS from inference speed
        inference_ms = self.speed.get("inference", 0) if hasattr(self, "speed") and self.speed else 0
        fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

        lines = []
        lines.append(sep)
        lines.append("Mamba-YOLO Instance Segmentation Evaluation Report")
        lines.append(sep)
        lines.append(f"Images evaluated:  {self.seen}")
        lines.append(f"Total instances:   {int(self.nt_per_class.sum())}")
        lines.append(f"Number of classes: {self.nc}")
        lines.append("")

        # --- Model efficiency ---
        lines.append("--- Model Efficiency ---")
        if n_params >= 0:
            lines.append(f"  Parameters:   {n_params / 1e6:.2f} M")
        else:
            lines.append(f"  Parameters:   N/A")
        if gflops >= 0:
            lines.append(f"  GFLOPs:       {gflops:.2f} G")
        else:
            lines.append(f"  GFLOPs:       N/A")
        if fps > 0:
            lines.append(f"  FPS:          {fps:.1f} (inference only)")
        else:
            lines.append(f"  FPS:          N/A")
        lines.append("")

        # Overall metrics
        lines.append("--- Overall Metrics ---")
        lines.append(f"{'Metric':<20} {'Box':>12} {'Mask':>12}")
        lines.append("-" * 44)
        lines.append(f"{'mAP@0.5':<20} {box.map50:>12.4f} {seg.map50:>12.4f}")
        lines.append(f"{'mAP@0.75':<20} {box.map75:>12.4f} {seg.map75:>12.4f}")
        lines.append(f"{'mAP@0.5:0.95':<20} {box.map:>12.4f} {seg.map:>12.4f}")
        lines.append(f"{'Precision':<20} {box.mp:>12.4f} {seg.mp:>12.4f}")
        lines.append(f"{'Recall':<20} {box.mr:>12.4f} {seg.mr:>12.4f}")
        lines.append(f"{'mIoU (union)':<20} {'-':>12} {np.mean(self.crack_mious) if self.crack_mious else 0.0:>12.4f}")
        lines.append(f"{'clDice':<20} {'-':>12} {np.mean(self.crack_cldices) if self.crack_cldices else 0.0:>12.4f}")
        lines.append(f"{'Mask fitness':<20} {'-':>12} {0.1 * seg.map50 + 0.9 * seg.map:>12.4f}")
        lines.append(f"{'Fitness':<20} {self.metrics.fitness:>12.4f}")
        lines.append("")

        # Per-class metrics
        if self.nc > 1 and len(self.metrics.ap_class_index) > 0:
            lines.append("--- Per-class Box Metrics ---")
            lines.append(
                f"{'Class':<20} {'Images':>8} {'Inst':>8} {'P':>8} {'R':>8} "
                f"{'mAP50':>8} {'mAP50-95':>10} {'mAP75':>8}"
            )
            lines.append("-" * 80)
            for i, c in enumerate(self.metrics.ap_class_index):
                bp, br, bap50, bap = self.metrics.box.class_result(i)
                bap75 = self.metrics.box.all_ap[i, 5] if len(self.metrics.box.all_ap) else 0.0
                name = self.names[c]
                lines.append(
                    f"{name:<20} {int(self.nt_per_image[c]):>8} {int(self.nt_per_class[c]):>8} "
                    f"{bp:>8.4f} {br:>8.4f} {bap50:>8.4f} {bap:>10.4f} {bap75:>8.4f}"
                )

            lines.append("")
            lines.append("--- Per-class Mask Metrics ---")
            lines.append(
                f"{'Class':<20} {'Images':>8} {'Inst':>8} {'P':>8} {'R':>8} "
                f"{'mAP50':>8} {'mAP50-95':>10} {'mAP75':>8}"
            )
            lines.append("-" * 80)
            for i, c in enumerate(self.metrics.ap_class_index):
                sp, sr, sap50, sap = self.metrics.seg.class_result(i)
                sap75 = self.metrics.seg.all_ap[i, 5] if len(self.metrics.seg.all_ap) else 0.0
                name = self.names[c]
                lines.append(
                    f"{name:<20} {int(self.nt_per_image[c]):>8} {int(self.nt_per_class[c]):>8} "
                    f"{sp:>8.4f} {sr:>8.4f} {sap50:>8.4f} {sap:>10.4f} {sap75:>8.4f}"
                )

        lines.append("")
        lines.append(sep)

        # Speed
        if hasattr(self, "speed") and self.speed:
            lines.append("--- Speed ---")
            lines.append(
                f"  Preprocess:  {self.speed.get('preprocess', 0):.1f}ms"
            )
            lines.append(
                f"  Inference:   {self.speed.get('inference', 0):.1f}ms"
            )
            lines.append(
                f"  Loss:        {self.speed.get('loss', 0):.1f}ms"
            )
            lines.append(
                f"  Postprocess: {self.speed.get('postprocess', 0):.1f}ms"
            )
            if fps > 0:
                lines.append(f"  FPS:         {fps:.1f}")
            lines.append("")

        # Model summary
        lines.append("--- Model Summary ---")
        if n_params >= 0:
            lines.append(f"  Parameters:   {n_params / 1e6:.2f} M ({n_params:,})")
        if gflops >= 0:
            lines.append(f"  GFLOPs:       {gflops:.2f} G")
        if n_params >= 0 and gflops >= 0:
            lines.append(f"  Params/FLOPs: {n_params / 1e6:.2f}M / {gflops:.2f}G")
        lines.append(sep)

        report_content = "\n".join(lines)
        with open(report_path, "w") as f:
            f.write(report_content)
        LOGGER.info(f"Detailed segmentation evaluation report saved to {report_path}")

    def eval_json(self, stats):
        """Return COCO-style object detection evaluation metrics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 5 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                        :2
                    ]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats
