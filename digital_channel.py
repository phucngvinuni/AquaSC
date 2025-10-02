
import numpy as np
import torch
import math

BITS_PER_SYMBOL = 4 

QAM16_CONSTELLATION = {
    (1,0,1,0): -3-3j, (1,0,1,1): -3-1j, (1,0,0,1): -3+1j, (1,0,0,0): -3+3j,
    (1,1,1,0): -1-3j, (1,1,1,1): -1-1j, (1,1,0,1): -1+1j, (1,1,0,0): -1+3j,
    (0,1,1,0):  1-3j, (0,1,1,1):  1-1j, (0,1,0,1):  1+1j, (0,1,0,0):  1+3j,
    (0,0,1,0):  3-3j, (0,0,1,1):  3-1j, (0,0,0,1):  3+1j, (0,0,0,0):  3+3j,
}
QAM16_DEMAPPING_TABLE = {v: k for k, v in QAM16_CONSTELLATION.items()}
QAM16_SYMBOLS = np.array(list(QAM16_DEMAPPING_TABLE.keys()))

def modulate_16qam(bits: np.ndarray) -> np.ndarray:
    """Điều chế chuỗi bit thành các symbol 16-QAM."""
    remainder = len(bits) % BITS_PER_SYMBOL
    if remainder != 0:
        bits = np.pad(bits, (0, BITS_PER_SYMBOL - remainder), 'constant', constant_values=0)
    
    symbols = []
    for i in range(0, len(bits), BITS_PER_SYMBOL):
        bit_chunk = tuple(bits[i:i+BITS_PER_SYMBOL])
        symbols.append(QAM16_CONSTELLATION[bit_chunk])
    return np.array(symbols)

def demodulate_16qam(noisy_symbols: np.ndarray) -> np.ndarray:
    """Giải điều chế các symbol 16-QAM nhiễu về chuỗi bit (hard decision)."""
    demod_bits = []
    for symbol in noisy_symbols:
        idx = np.argmin(np.abs(QAM16_SYMBOLS - symbol))
        closest_symbol = QAM16_SYMBOLS[idx]
        demod_bits.extend(QAM16_DEMAPPING_TABLE[closest_symbol])
    return np.array(demod_bits, dtype=int)

def simulate_rayleigh_channel(symbols: np.ndarray, snr_db: float) -> np.ndarray:
    """Mô phỏng kênh Rayleigh phẳng."""
    h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    signal_power = np.mean(np.abs(symbols)**2)
    if signal_power == 0: return symbols 
    symbols_normalized = symbols / np.sqrt(signal_power)

    snr_linear = 10**(snr_db / 10.0)
    noise_variance = 1 / snr_linear
    
    noise = np.sqrt(noise_variance / 2) * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
    
    received_signal = symbols_normalized * h + noise
    
    equalized_signal = received_signal / (h + 1e-8)
    return equalized_signal * np.sqrt(signal_power)

def bitstream_to_indices(bitstream: np.ndarray, bits_map: np.ndarray, num_indices: int) -> np.ndarray:
    """Chuyển đổi chuỗi bit về lại mảng chỉ số, với số bit thay đổi cho mỗi chỉ số."""
    indices = []
    current_pos = 0
    for i in range(num_indices):
        num_bits = int(bits_map[i])
        if current_pos + num_bits <= len(bitstream):
            bit_chunk_str = "".join(map(str, bitstream[current_pos : current_pos + num_bits]))
            try:
                indices.append(int(bit_chunk_str, 2))
            except ValueError:
                indices.append(0)
            current_pos += num_bits
        else:
            indices.append(0)
    return np.array(indices)


def transmit_and_receive_indices_batch(indices_tensor: torch.Tensor, vq_bits_map: torch.Tensor, snr_db: float):
    """
    Mô phỏng pipeline truyền kỹ thuật số cho một batch (KHÔNG có mã hóa kênh).
    """
    device = indices_tensor.device
    batch_size, num_patches = indices_tensor.shape
    
    indices_np = indices_tensor.cpu().numpy()
    vq_bits_np = vq_bits_map.cpu().numpy()
    
    recovered_indices_batch_list = []
    
    for i in range(batch_size):

        bitstream_list = [format(idx, f'0{int(bits)}b') for idx, bits in zip(indices_np[i], vq_bits_np[i])]
        source_bitstream_str = "".join(bitstream_list)
        source_bitstream = np.array([int(b) for b in source_bitstream_str], dtype=int)
        modulated_symbols = modulate_16qam(source_bitstream)
        received_symbols = simulate_rayleigh_channel(modulated_symbols, snr_db)
        recovered_bits = demodulate_16qam(received_symbols)
        if len(recovered_bits) > len(source_bitstream):
            recovered_bits = recovered_bits[:len(source_bitstream)]

        recovered_indices_np = bitstream_to_indices(recovered_bits, vq_bits_np[i], num_patches)
        recovered_indices_batch_list.append(torch.from_numpy(recovered_indices_np).long())

    return torch.stack(recovered_indices_batch_list).to(device)
