# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
对图片、视频、目录、通配路径、摄像头/网络流等做目标检测推理。

入口：parse_opt() 读命令行 → main() 做依赖检查与分支 → run() 加载模型、读数据、前向、NMS、按需保存。

框格式：模型输出多为 xyxy（像素，左上-右下）；写 txt 时常转为归一化 xywh（中心+宽高，相对原图宽高）。
NMS：按 conf 过滤重叠框，iou_thres 控制「多近算重复」；agnostic_nms 则类间也互相抑制（少见场景）。
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import xml.etree.ElementTree as ET  # VOC XML 手写序列化，无第三方依赖

# 本文件所在目录当作项目根，保证能 import 同仓库下的 models/utils
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对 cwd 的路径，便于打印/拼默认权重路径

from utils.env_config import get_data_yaml, get_device, get_output_dir, get_video_dir, get_weights

# 第三方：画框与调色；其余为仓库内模块（多后端推理、数据加载、NMS 与设备选择）
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.ddp_batch_buffer import (expected_batch_size, is_batch_boundary, resolve_batch_slot,
                                    should_rank_process_frame, valid_payloads_for_batch)
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.voc_incremental import filter_new_videos, mark_video_converted, state_path, write_source_list_txt

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def initialize_parallel_inference(device):
    """Resolve per-rank device binding for multi-process NPU inference."""
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        raise RuntimeError('NPU parallel inference requires torch.npu.is_available() == True')
    if LOCAL_RANK == -1:
        raise RuntimeError('--ddp-infer requires launch via torch.distributed.run')
    if not hasattr(dist, 'is_hccl_available') or not dist.is_hccl_available():
        raise RuntimeError('NPU parallel inference requires HCCL support')
    if not dist.is_initialized():
        dist.init_process_group(backend='hccl')
    torch.npu.set_device(LOCAL_RANK)
    resolved = torch.device('npu', LOCAL_RANK)
    return resolved


def should_process_frame(frame_index, *, ddp_infer, rank, world_size):
    """Return True when the current rank owns the frame under modulo sharding."""
    return not ddp_infer or world_size <= 1 or frame_index % world_size == rank


def summarize_parallel_counts(rank_frame_counts):
    pairs = [f'{rank}:{count}' for rank, count in enumerate(rank_frame_counts)]
    return ','.join(pairs)


def format_video_start_log(path, file_index, file_total, vid_cap):
    """Build a one-line summary when inference switches to a new video source."""
    resolved = Path(path).resolve()
    parts = [f"Start video {file_index}/{file_total}: {resolved.name}", f"path={resolved}"]
    if vid_cap is not None:
        frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(vid_cap.get(cv2.CAP_PROP_FPS) or 0)
        width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        parts.append(f"frames={frames if frames > 0 else 'unknown'}")
        parts.append(f"fps={fps:.2f}" if fps > 0 else "fps=unknown")
        parts.append(f"size={width}x{height}" if width > 0 and height > 0 else "size=unknown")
    return ", ".join(parts)


def normalize_source_for_detect(source):
    """Expand a plain directory source into a recursive glob for detect.py only."""
    source = str(source)
    if any(token in source for token in ('*', '?')):
        return source
    path = Path(source)
    if path.is_dir():
        return str(path / '**' / '*.*')
    return source


def save_voc_xml(save_dir, img_filename, img_width, img_height, detections, depth=3):
    """把一帧的框写成 PASCAL VOC XML；文件名与 jpg stem 一致，便于与 images/ 配对。

    Args:
        save_dir: annotations 目录。
        img_filename: 对应图片名（含 .jpg），写入 <filename>。
        img_width, img_height: 原图尺寸，写入 <size>。
        detections: [(x1,y1,x2,y2, class_name), ...]，像素整数框。
        depth: 通道数，彩色图 3。
    """
    annotation = ET.Element("annotation")
    
    # folder
    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"
    
    # filename
    filename = ET.SubElement(annotation, "filename")
    filename.text = img_filename
    
    # source (可选)
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    
    # size
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_width)
    height = ET.SubElement(size, "height")
    height.text = str(img_height)
    depth_elem = ET.SubElement(size, "depth")
    depth_elem.text = str(depth)
    
    # segmented (默认 0)
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    
    # 每个检测框
    for x1, y1, x2, y2, class_name in detections:
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = class_name
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(x1))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(y1))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(x2))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(y2))
    
    # 写入文件
    xml_path = save_dir / f"{Path(img_filename).stem}.xml"
    tree = ET.ElementTree(annotation)
    tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)


def save_voc_frame_and_xml(image_dir, annotations_dir, base_name, frame_image, detections):
    """Save the raw video frame plus its VOC XML sidecar.

    Args:
        image_dir: Target VOC images directory.
        annotations_dir: Target VOC annotations directory.
        base_name: Shared file stem for the JPG/XML pair.
        frame_image: Original BGR frame before any boxes or labels are drawn.
        detections: [(x1, y1, x2, y2, class_name), ...] in pixel coordinates.
    """
    img_save_path = image_dir / f"{base_name}.jpg"
    h, w = frame_image.shape[:2]
    cv2.imwrite(str(img_save_path), frame_image)
    save_voc_xml(annotations_dir, f"{base_name}.jpg", w, h, detections, depth=3)


def append_prediction_csv(csv_path, image_name, prediction, confidence):
    data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not csv_path.is_file():
            writer.writeheader()
        writer.writerow(data)


def summarize_detection_log(prefix, im_shape_hw, det, names):
    """Build the per-frame detection summary line before timing is appended."""
    summary = prefix + '%gx%g ' % im_shape_hw
    if len(det):
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            summary += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
    else:
        summary += '(no detections), '
    return summary


def serialize_detections(det):
    """Convert a detection tensor into a CPU-serializable list."""
    if not len(det):
        return []
    return det.detach().cpu().tolist()


def deserialize_detections(det_rows):
    """Rebuild a CPU tensor from serialized detection rows."""
    if not det_rows:
        return torch.zeros((0, 6), dtype=torch.float32)
    tensor = torch.tensor(det_rows, dtype=torch.float32)
    return tensor.view(-1, 6)


def ensure_batch_buffer_supported(*, batch_buffer, ddp_infer, buffer_size, world_size, source, is_file, is_url, webcam,
                                  screenshot, dataset):
    """Validate that batch-buffer mode is only used for the supported Phase 1 input shape."""
    if not batch_buffer:
        return
    if not ddp_infer:
        raise ValueError('--batch-buffer requires --ddp-infer')
    if world_size <= 1:
        raise ValueError('--batch-buffer requires WORLD_SIZE > 1')
    if buffer_size <= 0 or buffer_size > world_size:
        raise ValueError(f'--buffer-size must satisfy 1 <= buffer_size <= WORLD_SIZE ({world_size})')
    if webcam or screenshot or is_url:
        raise ValueError('--batch-buffer currently supports only local video files; online streams are Phase 2')
    files = [Path(path) for path in getattr(dataset, 'files', [])]
    video_flags = list(getattr(dataset, 'video_flag', []))
    if not files or len(files) != len(video_flags):
        raise ValueError('--batch-buffer requires a local video source resolvable by LoadImages')
    if not all(path.is_file() for path in files):
        raise ValueError('--batch-buffer requires every resolved source item to be a local file')
    if not all(flag for flag in video_flags):
        raise ValueError('--batch-buffer requires every resolved source item to be a video file')
    total_frames = int(getattr(dataset, 'frames', 0) or 0)
    if total_frames <= 0:
        raise ValueError('--batch-buffer requires a video source with a known positive frame count')


def render_frame_result(*,
                        det_rows,
                        path,
                        im0,
                        frame,
                        dataset_mode,
                        vid_cap,
                        vid_index,
                        vid_path,
                        vid_writer,
                        save_dir,
                        csv_path,
                        names,
                        line_thickness,
                        hide_labels,
                        hide_conf,
                        save_csv,
                        save_txt,
                        save_conf,
                        save_crop,
                        save_img_frames,
                        save_img,
                        view_img,
                        windows,
                        image_dir=None,
                        annotations_dir=None):
    """Apply save/render side effects for one frame using already-scaled detections."""
    det = deserialize_detections(det_rows)
    p = Path(path)
    save_path = str(save_dir / p.name)
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset_mode == 'image' else f'_{frame}')
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
    imc = im0.copy() if save_crop else im0
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

    if len(det):
        if save_img_frames:
            vid_tag = f"{p.stem}_{p.suffix[1:].lower()}" if p.suffix else p.stem
            base_name = f"{vid_tag}_frame{frame:06d}"
            detections_for_xml = []
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                class_name = names[int(cls)]
                detections_for_xml.append((x1, y1, x2, y2, class_name))
            save_voc_frame_and_xml(
                image_dir=image_dir,
                annotations_dir=annotations_dir,
                base_name=base_name,
                frame_image=im0.copy(),
                detections=detections_for_xml,
            )

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = names[c] if hide_conf else f'{names[c]}'
            confidence_str = f'{conf:.2f}'

            if save_csv:
                append_prediction_csv(csv_path, p.name, label, confidence_str)

            if save_txt:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                with open(f'{txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_img_frames or save_img or save_crop or view_img:
                box_label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, box_label, color=colors(c, True))
            if save_crop:
                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

    im0 = annotator.result()
    if view_img:
        if platform.system() == 'Linux' and p not in windows:
            windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)

    if save_img:
        if dataset_mode == 'image':
            cv2.imwrite(save_path, im0)
        else:
            if vid_path[vid_index] != save_path:
                vid_path[vid_index] = save_path
                if isinstance(vid_writer[vid_index], cv2.VideoWriter):
                    vid_writer[vid_index].release()
                if vid_cap:
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))
                vid_writer[vid_index] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[vid_index].write(im0)

@smart_inference_mode()
def run(
        weights,
        source,
        data,
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=True,
        save_csv=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=get_output_dir('detect'),
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        save_img_frames=False,
        vid_stride=1,
        voc_root=None,
        incremental_mark_voc_root=None,
        ddp_infer=False,
        batch_buffer=False,
        buffer_size=0,
        frame_shard_mode='mod',
        save_summary_only=False,
):
    """单轮推理：建 save_dir → 加载权重与 names → 选 Load* → 逐条 source 预处理/推理/NMS → 保存。

    常用开关：conf/iou 调检出敏感度；save_txt 写 YOLO 行标注；save_img_frames 抽帧+VOC XML；
    incremental_mark_voc_root 与 utils.voc_incremental 配合，在批量视频间写「已转换」状态。
    """
    source = normalize_source_for_detect(source)

    # ===================== 无扩展名/写错扩展名时：同 stem 下试常见视频后缀，避免找不到文件 =====================
    if '*' not in source and '?' not in source and not Path(source).exists():
        video_exts = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'mpeg', 'mpg']
        folder = Path(source).parent
        name = Path(source).stem
        for ext in video_exts:
            test_path = folder / f"{name}.{ext}"
            if test_path.exists():
                source = str(test_path)
                break
    # ==========================================================================================

    if save_summary_only:
        nosave = True
        save_txt = False
        save_csv = False
        save_conf = False
        save_crop = False
        save_img_frames = False

    # source 为 *.txt：每行一个路径，批跑；此时默认关 save_img，避免每条都落盘大图/视频占满磁盘；需要结果用 --save-txt 等。
    save_img = not nosave and not source.endswith('.txt')
    # 以下四类互斥描述「数据从哪来」，决定后面用 LoadImages / LoadStreams / LoadScreenshots
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 单张图或单个视频文件
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)  # 摄像头索引或直播流（非单文件 URL）
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # 远程单文件：下载/缓存到本地再推理

    # 输出根目录；exist_ok 时可复用同一 exp 目录做追加（与 VOC 抽帧并存时注意别混批次）
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    if save_txt:
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
    elif save_img or save_csv or save_img_frames:
        save_dir.mkdir(parents=True, exist_ok=True)

    if save_img_frames:
        # voc_root: 固定 VOC 数据集根目录，images/ 与 annotations/ 始终写入此处，便于多批次、多视频追加
        if voc_root:
            voc_base = Path(voc_root).resolve()
            image_dir = voc_base / 'images'
            annotations_dir = voc_base / 'annotations'
        else:
            image_dir = save_dir / 'images'
            annotations_dir = save_dir / 'annotations'
        image_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        if voc_root:
            LOGGER.info(f'VOC frames/labels append to: {image_dir} , {annotations_dir}')

    # 权重可 pt/onnx/engine 等；data.yaml 提供类别名与部分后端所需元数据；half 在 GPU 上省显存、略损精度
    effective_buffer_size = int(buffer_size or WORLD_SIZE)

    if ddp_infer:
        assert frame_shard_mode == 'mod', f"Unsupported --frame-shard-mode {frame_shard_mode}"
        device = initialize_parallel_inference(device)
        LOGGER.info(
            f'INFER init: rank={RANK} local_rank={LOCAL_RANK} world_size={WORLD_SIZE} '
            f'device={device} source={Path(source).resolve()} shard_mode={frame_shard_mode} '
            f'infer_mode={"batch_buffer" if batch_buffer else "mod"} '
            f'buffer_size={effective_buffer_size if batch_buffer else 0}'
        )
    else:
        device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt  # stride 决定输入边长需为 stride 倍数；names 即 cls id → 类名
    imgsz = check_img_size(imgsz, s=stride)

    LOGGER.warning(f"Loaded names: {names}")
    # print(f"Loaded names: {names}", flush=True)

    bs = 1  # 本脚本推理侧 batch；多路流时 bs = 流路数，每路仍逐帧处理
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        # 目录/通配/单文件/图片列表等：统一走 LoadImages（mode 会是 image/video）
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    ensure_batch_buffer_supported(
        batch_buffer=batch_buffer,
        ddp_infer=ddp_infer,
        buffer_size=effective_buffer_size,
        world_size=WORLD_SIZE,
        source=source,
        is_file=is_file,
        is_url=is_url,
        webcam=webcam,
        screenshot=screenshot,
        dataset=dataset,
    )
    if not webcam and not screenshot:
        source_path = Path(source)
        if source.endswith('.txt'):
            LOGGER.info(f'Source list mode: {source}')
        elif '*' in source:
            LOGGER.info(f"Source glob mode: {'recursive' if '**' in source else 'non-recursive'} pattern={source}")
        elif source_path.is_dir():
            LOGGER.info(f'Source directory mode: recursive dir={source_path.resolve()}')
        elif source_path.is_file():
            LOGGER.info(f'Source file mode: single path={source_path.resolve()}')
    # 每路流一个输出路径 + VideoWriter；非视频模式后面几乎不用
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 首次空推，触发 cudnn 等优化，避免把第一帧真实计时算进 benchmark
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    processed_frames = 0
    processed_batches = 0
    tail_batch_size = 0
    prev_video_for_mark = None
    logged_video_path = None
    needs_image_payload = save_img or save_crop or view_img or save_img_frames
    total_video_frames = int(getattr(dataset, 'frames', 0) or 0) if batch_buffer else 0
    csv_path = save_dir / 'predictions.csv'
    for path, im, im0s, vid_cap, s in dataset:
        # 增量模式且 source 为路径列表：每换一条视频文件，把「上一条」标为已处理，防中断丢状态
        if incremental_mark_voc_root is not None and dataset.mode == 'video':
            curv = str(Path(path).resolve())
            if prev_video_for_mark is not None and curv != prev_video_for_mark:
                mark_video_converted(incremental_mark_voc_root, Path(prev_video_for_mark))
            prev_video_for_mark = curv
        if dataset.mode == 'video':
            current_video_path = str(Path(path).resolve())
            if current_video_path != logged_video_path:
                LOGGER.info(format_video_start_log(path, dataset.count + 1, dataset.nf, vid_cap))
                logged_video_path = current_video_path
        shard_frame = dataset.count if webcam else getattr(dataset, 'frame', 0)
        frame_idx = max(shard_frame - 1, 0) if batch_buffer else shard_frame
        if batch_buffer:
            slot = resolve_batch_slot(frame_idx, effective_buffer_size)
            local_payload = None
            if should_rank_process_frame(frame_idx, RANK, effective_buffer_size):
                processed_frames += 1
                with dt[0]:
                    im_tensor = torch.from_numpy(im).to(model.device)
                    im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
                    im_tensor /= 255
                    if len(im_tensor.shape) == 3:
                        im_tensor = im_tensor[None]

                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im_tensor, augment=augment, visualize=visualize)

                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                det = pred[0]
                seen += 1
                if len(det):
                    det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0s.shape).round()
                log_s = summarize_detection_log(s, im_tensor.shape[2:], det, names)
                local_payload = {
                    'frame_idx': frame_idx,
                    'batch_id': slot.batch_id,
                    'path': path,
                    'frame': shard_frame,
                    'im0': im0s.copy() if needs_image_payload else None,
                    'im0_shape': tuple(im0s.shape),
                    'det_rows': serialize_detections(det),
                    'log_s': log_s,
                    'infer_time_ms': dt[1].dt * 1E3,
                }

            if is_batch_boundary(frame_idx, total_video_frames, effective_buffer_size):
                gathered_payloads = [None for _ in range(WORLD_SIZE)]
                dist.all_gather_object(gathered_payloads, local_payload)
                processed_batches += 1
                expected_size = expected_batch_size(slot.batch_id, total_video_frames, effective_buffer_size)
                if expected_size < effective_buffer_size:
                    tail_batch_size = expected_size
                if RANK == 0:
                    for payload in valid_payloads_for_batch(
                            gathered_payloads,
                            batch_id=slot.batch_id,
                            total_frames=total_video_frames,
                            buffer_size=effective_buffer_size):
                        render_frame_result(
                            det_rows=payload['det_rows'],
                            path=payload['path'],
                            im0=payload['im0'] if payload['im0'] is not None else im0s.copy(),
                            frame=payload['frame'],
                            dataset_mode=dataset.mode,
                            vid_cap=vid_cap,
                            vid_index=0,
                            vid_path=vid_path,
                            vid_writer=vid_writer,
                            save_dir=save_dir,
                            csv_path=csv_path,
                            names=names,
                            line_thickness=line_thickness,
                            hide_labels=hide_labels,
                            hide_conf=hide_conf,
                            save_csv=save_csv,
                            save_txt=save_txt,
                            save_conf=save_conf,
                            save_crop=save_crop,
                            save_img_frames=save_img_frames,
                            save_img=save_img,
                            view_img=view_img,
                            windows=windows,
                            image_dir=image_dir if save_img_frames else None,
                            annotations_dir=annotations_dir if save_img_frames else None,
                        )
                        LOGGER.info(f"{payload['log_s']}{payload['infer_time_ms']:.1f}ms")
            continue
        if not should_process_frame(shard_frame, ddp_infer=ddp_infer, rank=RANK, world_size=WORLD_SIZE):
            continue
        processed_frames += 1
        with dt[0]:
            # im:  letterbox 后的 NHWC numpy → NCHW tensor，/255 与训练分布一致
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]  # 补 batch 维：(C,H,W) → (1,C,H,W)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)  # 多尺度 TTA 用 augment；visualize 写中间特征图路径

        # NMS：同类框 IoU 高只保留高 conf；classes 非空时只保留指定类
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # pred 与 batch 维对齐：webcam 多路时 i 为第几路；单文件/图集时通常只有一个元素
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            summary_s = summarize_detection_log(s, im.shape[2:], det, names)
            if len(det):
                # 框从 letterbox 输入尺寸映射回原图 im0；之后均为原图像素 xyxy
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            render_frame_result(
                det_rows=serialize_detections(det),
                path=str(p),
                im0=im0,
                frame=frame,
                dataset_mode=dataset.mode,
                vid_cap=vid_cap,
                vid_index=i,
                vid_path=vid_path,
                vid_writer=vid_writer,
                save_dir=save_dir,
                csv_path=csv_path,
                names=names,
                line_thickness=line_thickness,
                hide_labels=hide_labels,
                hide_conf=hide_conf,
                save_csv=save_csv,
                save_txt=save_txt,
                save_conf=save_conf,
                save_crop=save_crop,
                save_img_frames=save_img_frames,
                save_img=save_img,
                view_img=view_img,
                windows=windows,
                image_dir=image_dir if save_img_frames else None,
                annotations_dir=annotations_dir if save_img_frames else None,
            )

        # 日志：本 batch 字符串 s + 本帧推理耗时（dt[1]）；无框时提示 no detections
        LOGGER.info(f"{summary_s}{dt[1].dt * 1E3:.1f}ms")

    # 循环正常结束：最后一条视频也要记入状态文件
    if incremental_mark_voc_root is not None and prev_video_for_mark is not None:
        mark_video_converted(incremental_mark_voc_root, Path(prev_video_for_mark))

    if ddp_infer:
        done_parts = [f'INFER done: rank={RANK}', f'processed_frames={processed_frames}']
        if batch_buffer:
            done_parts.extend([
                f'processed_batches={processed_batches}',
                'infer_mode=batch_buffer',
                f'buffer_size={effective_buffer_size}',
            ])
        LOGGER.info(' '.join(done_parts))
        if dist.is_initialized():
            rank_counts = [0 for _ in range(WORLD_SIZE)]
            dist.all_gather_object(rank_counts, processed_frames)
            aggregate_frames = sum(int(count) for count in rank_counts)
            if RANK == 0:
                aggregate_parts = [
                    f'INFER aggregate: world_size={WORLD_SIZE}',
                    f'rank_frame_counts={summarize_parallel_counts(rank_counts)}',
                    f'aggregate_frames={aggregate_frames}',
                ]
                if batch_buffer:
                    aggregate_parts.extend([
                        'infer_mode=batch_buffer',
                        f'buffer_size={effective_buffer_size}',
                        f'batch_count={processed_batches}',
                        f'tail_batch_size={tail_batch_size}',
                    ])
                aggregate_parts.append('parallel_infer_confirmed=true')
                LOGGER.info(' '.join(aggregate_parts))
            dist.barrier()
            dist.destroy_process_group()

    # 三段耗时按已处理帧数平均：预处理 / 推理 / NMS
    effective_seen = seen or 1
    t = tuple(x.t / effective_seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # --update：去掉优化器状态，便于只发布推理用权重


def parse_opt():
    """命令行参数：与 run() 形参一一对应；布尔类多为 store_true，出现即开启。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=get_weights(), help='model path')
    parser.add_argument('--source', type=str, default=get_video_dir(), help='file path')
    parser.add_argument('--data', type=str, default=get_data_yaml(), help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.25)  # 置信度阈值：越高框越少
    parser.add_argument('--iou-thres', type=float, default=0.45)  # NMS IoU：越高越保留邻近框
    parser.add_argument('--device', default=get_device(),
                        help='compute device, e.g. cpu, 0, 0,1, npu:0, npu:0,1,2,3')
    parser.add_argument('--view-img', action='store_true', help='show live preview windows (cv2.imshow); off by default')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-csv', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--save-crop', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')  # 类间一起做 NMS，默认按类分别抑制
    parser.add_argument('--augment', action='store_true')  # 推理 TTA，慢换略稳
    parser.add_argument('--visualize', action='store_true')  # 导出中间层特征图路径（调试用）
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--project', default=get_output_dir('detect'))
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true',
                        help='复用同一 save 目录（不创建 exp2/exp3）。与 --save-img-frames 联用时，帧与 XML 写入同一 runs/.../images|annotations 下追加')
    parser.add_argument('--line-thickness', default=3)
    parser.add_argument('--hide-labels', default=False, action='store_true')
    parser.add_argument('--hide-conf', default=False, action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--vid-stride', type=int, default=1)  # 视频隔几帧取 1 帧，加快扫盘/降采样
    parser.add_argument('--save-img-frames', action='store_true', dest='save_img_frames',
                        help='save annotated video frames as jpg images')
    parser.add_argument('--voc-root', type=str, default='',
                        help='VOC 输出根目录：与 --save-img-frames 同时使用时，将 JPG 写入 {voc-root}/images、XML 写入 {voc-root}/annotations（目录已存在则继续追加）。不设则仍写在 {save_dir}/images 与 annotations')
    parser.add_argument('--incremental-mp4', action='store_true', dest='incremental_mp4',
                        help='递归扫描 --source 下与 YOLOv5 一致的视频后缀(asf/avi/gif/m4v/mkv/mov/mp4/mpeg/mpg/ts/wmv，不区分大小写)，跳过已在 --voc-root 状态文件中记录的文件，仅转换新文件；须与 --voc-root、--save-img-frames 同用')
    parser.add_argument('--ddp-infer', action='store_true',
                        help='通过 torch.distributed.run 启动多进程 NPU 并行推理，按 frame % world_size 分片')
    parser.add_argument('--batch-buffer', action='store_true',
                        help='启用 DDP 批缓冲推理（Phase 1 仅支持单个本地视频文件）')
    parser.add_argument('--buffer-size', type=int, default=0,
                        help='批缓冲大小，默认等于 WORLD_SIZE，且必须满足 1 <= buffer_size <= WORLD_SIZE')
    parser.add_argument('--frame-shard-mode', type=str, default='mod',
                        help='frame sharding mode for --ddp-infer, currently only supports mod')
    parser.add_argument('--save-summary-only', action='store_true',
                        help='只保留日志/摘要，不保存图片、视频、txt、csv 或 crop 输出')

    opt = parser.parse_args()
    opt.voc_root = opt.voc_root.strip() or None
    # 只给一个边长时扩成 [h,w] 两个相同值，与训练侧 imgsz 约定一致
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    """非增量：直接 run(opt)。增量：筛出新 mp4 → 写临时路径列表 txt → run 且传入 incremental_mark_voc_root。"""
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    if opt.incremental_mp4:
        if not opt.voc_root or not opt.save_img_frames:
            LOGGER.error('--incremental-mp4 需要同时指定 --voc-root 与 --save-img-frames')
            raise SystemExit(2)
        scan_dir = Path(opt.source).resolve()
        if not scan_dir.is_dir():
            LOGGER.error('--incremental-mp4 要求 --source 为已存在的目录')
            raise SystemExit(2)
        voc_base = Path(opt.voc_root).resolve()
        new_videos = filter_new_videos(scan_dir, voc_base)
        if not new_videos:
            LOGGER.info(f'Incremental video: 无新文件，目录 {scan_dir}，状态 {state_path(voc_base)}')
            return
        LOGGER.info(f'Incremental video: 本次转换 {len(new_videos)} 个新视频 -> {voc_base}')
        tmp = write_source_list_txt(new_videos)  # YOLO 认 *.txt 为路径列表；用完删除临时文件
        try:
            kw = vars(opt).copy()
            kw.pop('incremental_mp4', None)
            kw['source'] = tmp
            kw['incremental_mark_voc_root'] = voc_base  # run 内按视频切换写状态，供下次 filter_new_videos 跳过
            run(**kw)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return
    kw = vars(opt).copy()
    kw.pop('incremental_mp4', None)  # run() 无此形参，避免 TypeError
    run(**kw)


if __name__ == '__main__':
    # 脚本直接运行时：解析 CLI → 检查依赖/分支 → 进入 run
    opt = parse_opt()
    main(opt)
