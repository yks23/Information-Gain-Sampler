#!/usr/bin/env python
"""
FID / sFID / IS / Precision / Recall 评测脚本

基于 OpenAI 的 Inception v3 评测方案，计算以下指标：
  - Inception Score (IS)
  - Fréchet Inception Distance (FID)
  - Spatial FID (sFID)
  - Precision & Recall

支持两种输入格式：
  1. .npz 文件（预先保存的参考数据，如 VIRTUAL_imagenet512.npz）
  2. 图像目录（自动递归扫描，支持 GenEval 目录结构）

用法：
  # 参考批次为 npz，样本批次为图像目录
  python eval_fid.py VIRTUAL_imagenet512.npz ./output_images --batch-size 64

  # 两个都是图像目录
  python eval_fid.py /path/to/ref_images /path/to/gen_images --batch-size 64

  # 将生成图像 resize 到 256x256 后再计算 FID
  python eval_fid.py VIRTUAL_imagenet256.npz ./output_512 --batch-size 64 --resize 256

依赖：
  pip install tensorflow scipy numpy tqdm Pillow
"""

import argparse
import io
import os
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple
import time

import numpy as np
import requests
from scipy import linalg
from tqdm.auto import tqdm

try:
    from PIL import Image
except ImportError:
    Image = None

import tensorflow.compat.v1 as tf


# ============================================================
# 常量
# ============================================================
INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = os.path.join(os.path.dirname(__file__), "classify_image_graph_def.pb")
FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"


# ============================================================
# FID 统计量
# ============================================================
class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma
        mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
        sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape
        assert sigma1.shape == sigma2.shape

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            warnings.warn(f"fid calculation produces singular product; adding {eps} to diagonal")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


# ============================================================
# Evaluator
# ============================================================
class Evaluator:
    def __init__(self, session, batch_size=64, softmax_batch_size=512, resize=None):
        self.sess = session
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self.resize = resize  # 目标分辨率，如 256 表示 resize 到 256x256
        self.manifold_estimator = ManifoldEstimator(session)
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)

    def warmup(self):
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))

    def read_activations(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if os.path.isdir(path):
            print(f"  从目录加载图像: {path}")
            image_batches = load_images_from_directory(path, self.batch_size, resize=self.resize)
            return self.compute_activations(image_batches)
        else:
            print(f"  从 npz 文件加载: {path}")
            with open_npz_array(path, "arr_0") as reader:
                return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(self, batches: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        preds, spatial_preds = [], []
        total_images = 0
        for batch in tqdm(batches, desc="  计算激活值", unit="batch"):
            batch = batch.astype(np.float32)
            total_images += batch.shape[0]
            pred, spatial_pred = self.sess.run(
                [self.pool_features, self.spatial_features], {self.image_input: batch}
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        print(f"  共处理 {total_images} 张图像")
        return np.concatenate(preds, axis=0), np.concatenate(spatial_preds, axis=0)

    def read_statistics(self, path, activations):
        if os.path.isdir(path):
            return tuple(self.compute_statistics(x) for x in activations)
        try:
            obj = np.load(path)
            if "mu" in list(obj.keys()):
                return FIDStatistics(obj["mu"], obj["sigma"]), FIDStatistics(obj["mu_s"], obj["sigma_s"])
        except Exception:
            pass
        return tuple(self.compute_statistics(x) for x in activations)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        return FIDStatistics(np.mean(activations, axis=0), np.cov(activations, rowvar=False))

    def compute_inception_score(self, activations: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i:i + self.softmax_batch_size]
            softmax_out.append(self.sess.run(self.softmax, feed_dict={self.softmax_input: acts}))
        preds = np.concatenate(softmax_out, axis=0)
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i:i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            scores.append(np.exp(np.mean(np.sum(kl, 1))))
        return float(np.mean(scores))

    def compute_prec_recall(self, activations_ref, activations_sample):
        radii_1 = self.manifold_estimator.manifold_radii(activations_ref)
        radii_2 = self.manifold_estimator.manifold_radii(activations_sample)
        pr = self.manifold_estimator.evaluate_pr(activations_ref, radii_1, activations_sample, radii_2)
        return float(pr[0][0]), float(pr[1][0])


# ============================================================
# ManifoldEstimator & DistanceBlock
# ============================================================
class ManifoldEstimator:
    def __init__(self, session, row_batch_size=10000, col_batch_size=10000,
                 nhood_sizes=(3,), clamp_to_percentile=None, eps=1e-5):
        self.distance_block = DistanceBlock(session)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def manifold_radii(self, features):
        num_images = len(features)
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)
        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]
            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                distance_batch[0:end1 - begin1, begin2:end2] = \
                    self.distance_block.pairwise_distances(row_batch, features[begin2:end2])
            radii[begin1:end1, :] = np.concatenate(
                [x[:, self.nhood_sizes] for x in _numpy_partition(distance_batch[0:end1 - begin1, :], seq, axis=1)],
                axis=0)
        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def evaluate_pr(self, features_1, radii_1, features_2, radii_2):
        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=bool)
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=bool)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2])
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )


class DistanceBlock:
    def __init__(self, session):
        self.session = session
        with session.graph.as_default():
            self._features_batch1 = tf.placeholder(tf.float32, shape=[None, None])
            self._features_batch2 = tf.placeholder(tf.float32, shape=[None, None])
            distance_block_16 = _batch_pairwise_distances(
                tf.cast(self._features_batch1, tf.float16),
                tf.cast(self._features_batch2, tf.float16))
            self.distance_block = tf.cond(
                tf.reduce_all(tf.math.is_finite(distance_block_16)),
                lambda: tf.cast(distance_block_16, tf.float32),
                lambda: _batch_pairwise_distances(self._features_batch1, self._features_batch2))
            self._radii1 = tf.placeholder(tf.float32, shape=[None, None])
            self._radii2 = tf.placeholder(tf.float32, shape=[None, None])
            dist32 = tf.cast(self.distance_block, tf.float32)[..., None]
            self._batch_1_in = tf.math.reduce_any(dist32 <= self._radii2, axis=1)
            self._batch_2_in = tf.math.reduce_any(dist32 <= self._radii1[:, None], axis=0)

    def pairwise_distances(self, U, V):
        return self.session.run(self.distance_block,
                                feed_dict={self._features_batch1: U, self._features_batch2: V})

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        return self.session.run(
            [self._batch_1_in, self._batch_2_in],
            feed_dict={self._features_batch1: batch_1, self._features_batch2: batch_2,
                       self._radii1: radii_1, self._radii2: radii_2})


# ============================================================
# TF 图构建辅助
# ============================================================
def _batch_pairwise_distances(U, V):
    with tf.variable_scope("pairwise_dist_block"):
        norm_u = tf.reshape(tf.reduce_sum(tf.square(U), 1), [-1, 1])
        norm_v = tf.reshape(tf.reduce_sum(tf.square(V), 1), [1, -1])
        return tf.maximum(norm_u - 2 * tf.matmul(U, V, False, True) + norm_v, 0.0)


def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    print("下载 InceptionV3 模型...")
    with requests.get(INCEPTION_V3_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)


def _create_feature_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    pool3, spatial = tf.import_graph_def(
        graph_def, input_map={f"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME], name=prefix)
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial


def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (matmul,) = tf.import_graph_def(
        graph_def, return_elements=["softmax/logits/MatMul"], name=prefix)
    w = matmul.inputs[1]
    return tf.nn.softmax(tf.matmul(input_batch, w))


def _update_shapes(pool3):
    ops = pool3.graph.get_operations()
    for op in ops:
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
                shape = [s for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3


# ============================================================
# NPZ 读取
# ============================================================
class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size): pass

    @abstractmethod
    def remaining(self): pass

    def read_batches(self, batch_size):
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None:
                    break
                yield batch
        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return BatchIterator(gen_fn, num_batches)


class BatchIterator:
    def __init__(self, gen_fn, length):
        self.gen_fn = gen_fn
        self.length = length
    def __len__(self): return self.length
    def __iter__(self): return self.gen_fn()


class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f, self.shape, self.dtype, self.idx = arr_f, shape, dtype, 0

    def read_batch(self, batch_size):
        if self.idx >= self.shape[0]:
            return None
        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs
        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)
        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])

    def remaining(self):
        return max(0, self.shape[0] - self.idx)


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr, self.idx = arr, 0

    @classmethod
    def load(cls, path, arr_name):
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size):
        if self.idx >= self.arr.shape[0]:
            return None
        res = self.arr[self.idx:self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self):
        return max(0, self.arr.shape[0] - self.idx)


@contextmanager
def open_npz_array(path, arr_name):
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)


def _read_bytes(fp, size, error_template="ran out of data"):
    data = bytes()
    while True:
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        raise ValueError(f"EOF: reading {error_template}, expected {size} bytes got {len(data)}")
    return data


@contextmanager
def _open_npy_file(path, arr_name):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist():
                raise ValueError(f"missing {arr_name} in npz file")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f:
                yield arr_f


def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers
    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx:start_idx + size])
        start_idx += size
    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))


# ============================================================
# 图像目录加载
# ============================================================
def collect_image_files(directory):
    """递归收集目录中的图像文件，支持 GenEval 目录结构（samples/ 子目录）"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    skip_dirs = {'entropy-vis', 'entropy_vis', 'vis', 'visualization', '.git', '__pycache__'}
    image_files = []
    seen_files = set()

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d.lower() not in skip_dirs]
        path_parts = root.split(os.sep)
        if any(any(sd in part.lower() for sd in skip_dirs) for part in path_parts):
            continue
        if 'samples' in dirs:
            samples_dir = os.path.join(root, 'samples')
            for file in os.listdir(samples_dir):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(samples_dir, file)
                    norm_path = os.path.normpath(full_path)
                    if norm_path not in seen_files:
                        image_files.append(full_path)
                        seen_files.add(norm_path)
        else:
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(root, file)
                    norm_path = os.path.normpath(full_path)
                    if norm_path not in seen_files:
                        image_files.append(full_path)
                        seen_files.add(norm_path)
    image_files.sort()
    return image_files


def load_images_from_directory(directory, batch_size, resize=None):
    """从目录流式加载图像，按批次返回 numpy 数组

    Args:
        directory: 图像目录路径
        batch_size: 每批图像数量
        resize: 可选，目标分辨率（int），如 256 表示 resize 到 256x256
    """
    image_files = collect_image_files(directory)
    if len(image_files) == 0:
        raise ValueError(f"在目录 {directory} 中未找到任何图像文件")
    print(f"  找到 {len(image_files)} 张图像")
    if resize is not None:
        print(f"  将 resize 到 {resize}x{resize}")

    current_batch = []
    first_shape = None
    total_loaded = 0

    for img_path in tqdm(image_files, desc="  加载图像", unit="img", leave=False):
        try:
            img = Image.open(img_path).convert('RGB')
            if resize is not None:
                img = img.resize((resize, resize), Image.LANCZOS)
            img_array = np.array(img, dtype=np.uint8)
            if first_shape is None:
                first_shape = img_array.shape
            elif img_array.shape != first_shape:
                continue
            current_batch.append(img_array)
            total_loaded += 1
            if len(current_batch) >= batch_size:
                yield np.stack(current_batch, axis=0)
                current_batch = []
        except Exception as e:
            warnings.warn(f"无法加载图像 {img_path}: {e}")

    if current_batch:
        yield np.stack(current_batch, axis=0)
    print(f"  成功加载 {total_loaded} 张图像")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="计算 FID / sFID / IS / Precision / Recall")
    parser.add_argument("ref_batch", help="参考批次: npz 文件路径或图像目录")
    parser.add_argument("sample_batch", help="样本批次: npz 文件路径或图像目录")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小 (默认 64)")
    parser.add_argument("--resize", type=int, default=None,
                        help="将样本图像 resize 到指定分辨率再计算 (如 --resize 256)")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 70)
    print("FID / IS / Precision / Recall 评测")
    print("=" * 70)
    print(f"参考批次: {args.ref_batch}")
    print(f"样本批次: {args.sample_batch}")
    if args.resize:
        print(f"Resize 到: {args.resize}x{args.resize}")

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    print("\n[1/6] 初始化 TensorFlow...")
    evaluator = Evaluator(tf.Session(config=config), batch_size=args.batch_size, resize=args.resize)

    print("[2/6] 预热...")
    evaluator.warmup()

    print("[3/6] 计算参考批次激活值...")
    ref_acts = evaluator.read_activations(args.ref_batch)

    print("[4/6] 计算参考批次统计信息...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

    print("[5/6] 计算样本批次激活值...")
    sample_acts = evaluator.read_activations(args.sample_batch)

    print("[6/6] 计算样本批次统计信息...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(args.sample_batch, sample_acts)

    # 计算指标
    print("\n" + "=" * 70)
    is_score = evaluator.compute_inception_score(sample_acts[0])
    fid_score = sample_stats.frechet_distance(ref_stats)
    sfid_score = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])

    print("评估结果")
    print("=" * 70)
    print(f"  Inception Score : {is_score:.4f}")
    print(f"  FID             : {fid_score:.4f}")
    print(f"  sFID            : {sfid_score:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {recall:.4f}")
    print("=" * 70)
    print(f"总耗时: {time.time() - start_time:.1f} 秒")


if __name__ == "__main__":
    main()

