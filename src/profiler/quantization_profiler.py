import faiss
import numpy as np
from models.patchcore.product_quantizer import ProductQuantizer


def compute_quantizer_config_size(quantizer: faiss.IndexPQ) -> int:
    centroids_size = quantizer.pq.centroids.size() * np.dtype(np.float32).itemsize
    m_size = np.dtype(np.int32).itemsize
    k_size = np.dtype(np.int32).itemsize
    total_size = centroids_size + m_size + k_size
    return total_size


def compute_product_quantization_efficiency(
    coreset: np.ndarray, compressed_coreset: np.ndarray, quantizer: ProductQuantizer
) -> tuple[float, np.float64]:
    np_array_type = coreset.dtype
    compressed_np_array_type = compressed_coreset.dtype
    original_shape = coreset.shape
    compressed_shape = compressed_coreset.shape
    product_quantized_config_size = compute_quantizer_config_size(quantizer.quantizer)
    original_bitrate = np_array_type.itemsize * np.prod(original_shape) * 8
    compressed_bitrate = (
        compressed_np_array_type.itemsize * np.prod(compressed_shape)
        + product_quantized_config_size
    ) * 8
    compression_efficiency = 1 - compressed_bitrate / original_bitrate
    dequantized_coreset = quantizer.decode(compressed_coreset).cpu().numpy()
    distortion = np.linalg.norm(coreset - dequantized_coreset) / np.linalg.norm(coreset)
    return compression_efficiency, distortion.astype(np.float64)
