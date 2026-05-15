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
import xml.etree.ElementTree as ET  # VOC XML 手写序列化，无第三方依赖

# 本文件所在目录当作项目根，保证能 import 同仓库下的 models/utils
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对 cwd 的路径，便于打印/拼默认权重路径

# 第三方：画框与调色；其余为仓库内模块（多后端推理、数据加载、NMS 与设备选择）
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.voc_incremental import filter_new_videos, mark_video_converted, state_path, write_source_list_txt

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
        project=ROOT / 'runs/detect',
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
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

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
    prev_video_for_mark = None
    logged_video_path = None
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

        # 可选：每张一行写入 CSV，便于后处理统计
        csv_path = save_dir / 'predictions.csv'
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # pred 与 batch 维对齐：webcam 多路时 i 为第几路；单文件/图集时通常只有一个元素
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            # 视频/流：同 stem 多帧，txt 用 stem_frame 区分；纯图片每文件一 txt
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # W,H,W,H：xywh 像素值除以对应宽高 → YOLO txt 用 0~1 相对坐标
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 框从 letterbox 输入尺寸映射回原图 im0；之后均为原图像素 xyxy
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 可选：强制把所有框类别改为某一类（如只标 drone）。默认关闭，避免误改类别。
                # det 每行: x1,y1,x2,y2, conf, cls
                # try:
                #     # 如果 names 是字典，那么遍历字典的键值对，找到值等于 'drone' 的那个键（即类别ID）
                #     if isinstance(names, dict):
                #         drone_class_id = next(k for k, v in names.items() if v == 'drone')
                #     else:
                #         # 如果 names 是列表，例如 ['person', 'drone', ...]
                #         # 直接返回 'drone' 在列表中的索引（即类别ID）
                #         drone_class_id = names.index('drone')
                # except (StopIteration, ValueError):
                #     LOGGER.warning("Class 'drone' not found in names, using original class ids")
                # else:
                #     print(f"Found drone class ID: {drone_class_id}")
                #     det[:, 5] = drone_class_id

                if save_img_frames:
                    # 使用「不含扩展名的文件名 + 扩展名」作前缀，降低不同视频同名 stem 冲突；多视频同目录一次跑也会区分
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

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence_str = f'{conf:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:
                        # YOLO txt：class + 归一化中心 xywh（相对整张图宽高）；save_conf 则多写一个 conf
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img_frames or save_img or save_crop or view_img:
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 把框画回 BGR 图；无检测时 annotator 等价原图
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # 落盘：图片写单张文件；视频则复用或新建 VideoWriter 连续 write
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 日志：本 batch 字符串 s + 本帧推理耗时（dt[1]）；无框时提示 no detections
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 循环正常结束：最后一条视频也要记入状态文件
    if incremental_mark_voc_root is not None and prev_video_for_mark is not None:
        mark_video_converted(incremental_mark_voc_root, Path(prev_video_for_mark))

    # 三段耗时按已处理帧数平均：预处理 / 推理 / NMS
    t = tuple(x.t / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # --update：去掉优化器状态，便于只发布推理用权重


def parse_opt():
    """命令行参数：与 run() 形参一一对应；布尔类多为 store_true，出现即开启。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'checkpoint/yolov5_best.pt', help='model path')
    parser.add_argument('--source', type=str, default='D:\\VideoData\\video\\laishui0511\\00000003468000000', help='file path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/dataAirVis.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.25)  # 置信度阈值：越高框越少
    parser.add_argument('--iou-thres', type=float, default=0.45)  # NMS IoU：越高越保留邻近框
    parser.add_argument('--device', default='0', help='cuda device 0')  # GPU
    parser.add_argument('--view-img', default=True, action='store_true', help='show results')  # 实时画面
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
    parser.add_argument('--project', default=ROOT / 'runs/detect')
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
