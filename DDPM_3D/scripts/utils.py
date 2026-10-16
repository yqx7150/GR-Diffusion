import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pydicom
import numpy as np
from glob import glob

import torch as th
import torch.distributed as dist

from guided_diffusion import logger
from icecream import ic

def get_SUV(path):
    ds = pydicom.dcmread(path)
    def dicom_hhmmss(t):    # dicom存时间格式:小时、分钟、秒(每个占两位),这里转化回秒
        t = str(t)
        if len(t) == 5:     # 有些提取时漏了个0，小时的位置只有一位，这里把零补上
            t = '0'+t
        h_t = float(t[0:2])
        m_t = float(t[2:4])
        s_t = float(t[4:6])
        return h_t*3600+m_t*60+s_t
    PET = ds.pixel_array
    RadiopharmaceuticalInformationSequence = ds.RadiopharmaceuticalInformationSequence[0]
    ST = ds.SeriesTime
    AT = ds.AcquisitionTime
    PW = ds.PatientWeight
    RST = str(RadiopharmaceuticalInformationSequence['RadiopharmaceuticalStartTime'].value)
    RTD = str(RadiopharmaceuticalInformationSequence['RadionuclideTotalDose'].value)
    RHL = str(RadiopharmaceuticalInformationSequence['RadionuclideHalfLife'].value)
    RS = ds.RescaleSlope
    RI = ds.RescaleIntercept
    CS = ds.DecayCorrection
    # [ST, AT, PW, RST, RTD, RHL, RS, RI] = data 		# AT基本等于ST,RI一般是0
    decay_time = dicom_hhmmss(ST)-dicom_hhmmss(RST)
    decay_dose = float(RTD) * pow(2, -float(decay_time) / float(6600))
    SUVbwScaleFactor = (1000 * float(PW)) / (decay_dose)
    PET_SUV = (PET * float(RS) + float(RI)) * SUVbwScaleFactor    	
    return PET_SUV


def load_data_for_worker_dcm360p(base_samples, batch_size, file_type="dcm"):
    low_pets = []
    if file_type == "dcm":
        low_pet_paths = sorted(glob(f"{base_samples}/*"))
        for low_pet_path in low_pet_paths:
            low_pet_suv = get_SUV(low_pet_path)[84: 276, 36: 324].astype(np.float32)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=2)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=0)
            low_pets.append(low_pet_suv)
    elif file_type == "npy":
        volumn = np.load(base_samples)
        num_slices = volumn.shape[0]
        for idx in range(num_slices):
            low_pet_suv = volumn[idx, :, :]
            low_pet_suv = low_pet_suv - low_pet_suv.min()
            low_pet_suv = low_pet_suv[84: 276, 36: 324].astype(np.float32)
            low_pet_suv = (low_pet_suv - low_pet_suv.min()) / (low_pet_suv.max() - low_pet_suv.min() + 1e-6)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=2)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=0)
            low_pets.append(low_pet_suv)

    # Divide the 3D PET volume of shape [192, 288, 520] into 6 overlapping patches along the axial axis.
    # Each patch has shape [192, 288, 96], with 10-pixel overlap between adjacent patches.
    # The final patch includes 16 pixels of overlap to cover the volume,
    # but only 10 of those overlapping pixels are used in the final reconstruction.
    image_arr = np.zeros((9,192,288,96), dtype=np.float32)
    index = 0
    for i in (0, 80, 160, 240, 320, 400, 480, 560, 577):
        image_arr[index, :, :, :] = np.concatenate(low_pets[i: i + 96], axis=3)
        index += 1

    # Preprocessing: convert to SUV (assumed already done), clip to 16, then normalize to [0, 1]
    if file_type == "dcm":
        image_arr[image_arr>16] = 16
        image_arr = image_arr/16


    buffer = []
    label_buffer = []
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])

            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).to(th.float32)
                batch = batch.unsqueeze(0) 
                batch = batch.permute(0, 1, 4, 2, 3)
                logger.log('batch_shape:{%d,%d,%d,%d,%d}' % (batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]))
                res = dict(low_res=batch)
                yield res
                buffer, label_buffer = [], []

def reverse_2_original_dimension_dcm360p(quadra_3D_864):
    # This function merges the 6 overlapping patches along the axial dimension to reconstruct
    # the original [192, 288, 520] PET volume. Overlapping areas are blended using weighted average.
    quadra_3D_673 = np.zeros((192, 288, 673))
    index_a = [0, 96, 176, 256, 336, 416, 496, 576, 656]
    index_b = [80, 160, 240, 320, 400, 480, 560, 640]

    # First non-overlapping segment
    quadra_3D_673[:, :, 0: 80] = quadra_3D_864[:, :, 0: 80]
    for i in range(0, 9):
        if i > 0 and i < 8:
            quadra_3D_673[:, :, index_a[i]: index_a[i]+64] = quadra_3D_864[:, :, 96*i+16: 96*i+80]
        elif i == 8:
            quadra_3D_673[:, :, index_a[i]:] = quadra_3D_864[:, :, 847:]
        
        if i < 7:
            for j in range(0, 16):
                quadra_3D_673[:, :, index_b[i]+j: index_b[i]+j+1] = quadra_3D_864[:, :, 96*i+80+j: 96*i+80+j+1] * (1-j/15)+ \
                                                                    quadra_3D_864[:, :, 96*(i+1)+j: 96*(i+1)+j+1] * (j/15)
        elif i == 7:
            for j in range(0, 16):
                quadra_3D_673[:, :, index_b[i]+j: index_b[i]+j+1] = quadra_3D_864[:, :, 96*i+80+j: 96*i+80+j+1] * (1-j/15)+ \
                                                                    quadra_3D_864[:, :, 96*(i+1)+j+63: 96*(i+1)+j+1+63] * (j/15)

    return quadra_3D_673

def postprocess(quadra_3D_673):
    # quadra_3D_673 -= quadra_3D_673.min()
    return np.transpose(quadra_3D_673, (2, 0 ,1))


def load_data_for_worker_dcm432x250x250(base_samples, batch_size, file_type="dcm"):
    low_pets = []
    if file_type == "dcm":
        low_pet_paths = sorted(glob(f"{base_samples}/*"))
        for low_pet_path in low_pet_paths:
            low_pet_suv = get_SUV(low_pet_path).astype(np.float32)
            low_pet_suv = low_pet_suv[29: 221, :]
            low_pet_suv = np.pad(low_pet_suv, pad_width=((0, 0), (19, 19)))
            low_pet_suv = np.expand_dims(low_pet_suv, axis=2)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=0)
            low_pets.append(low_pet_suv)
    elif file_type == "npy":
        volumn = np.load(base_samples)
        num_slices = volumn.shape[0]
        for idx in range(num_slices):
            low_pet_suv = volumn[idx, :, :]
            low_pet_suv = low_pet_suv - low_pet_suv.min()
            low_pet_suv = low_pet_suv.astype(np.float32)
            low_pet_suv = check_shape_and_crop(low_pet_suv, (192, 288))
            assert low_pet_suv.shape == (192, 288)
            low_pet_suv = (low_pet_suv - low_pet_suv.min()) / (low_pet_suv.max() - low_pet_suv.min() + 1e-6)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=2)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=0)
            low_pets.append(low_pet_suv)

    # Divide the 3D PET volume of shape [192, 288, 520] into 6 overlapping patches along the axial axis.
    # Each patch has shape [192, 288, 96], with 10-pixel overlap between adjacent patches.
    # The final patch includes 16 pixels of overlap to cover the volume,
    # but only 10 of those overlapping pixels are used in the final reconstruction.
    image_arr = np.zeros((6,192,288,82), dtype=np.float32)
    index = 0
    for i in (0, 72, 144, 216, 288, 350):
        image_arr[index, :, :, :] = np.concatenate(low_pets[i: i + 82], axis=3)
        index += 1

    # Preprocessing: convert to SUV (assumed already done), clip to 16, then normalize to [0, 1]
    if file_type == "dcm":
        image_arr[image_arr>16] = 16
        image_arr = image_arr/16


    buffer = []
    label_buffer = []
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])

            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).to(th.float32)
                batch = batch.unsqueeze(0) 
                batch = batch.permute(0, 1, 4, 2, 3)
                logger.log('batch_shape:{%d,%d,%d,%d,%d}' % (batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]))
                res = dict(low_res=batch)
                yield res
                buffer, label_buffer = [], []

def reverse_2_original_dimension_432x250x250(blocks_3D_492):
    """
    将形状为(192, 288, 492)的6个块拼接数组重建为(192, 288, 432)
    6个块，每块82层，重叠10层
    块在原始体积中的起始索引: [0, 72, 144, 216, 288, 350]
    """
    # 输入: (192, 288, 492) = 6个块 * 82层
    # 输出: (192, 288, 432)
    
    reconstructed = np.zeros((192, 288, 432), dtype=np.float32)
    
    # 定义块在拼接数组中的起始索引(深度维度)
    block_starts = [0, 82, 164, 246, 328, 410]  # 每个块在拼接数组中的起始位置
    block_orig_starts = [0, 72, 144, 216, 288, 350]  # 每个块在原始体积中的起始位置
    
    # 1. 处理第一个块的前72层（非重叠部分）
    reconstructed[:, :, 0:72] = blocks_3D_492[:, :, 0:72]
    
    # 2. 处理最后一个块的后62层（非重叠部分）
    # 最后一个块(索引350)的20-82层对应原始370-432层
    reconstructed[:, :, 370:432] = blocks_3D_492[:, :, 430:492]
    
    # 3. 处理中间4个10层重叠区域
    for i in range(4):  # 处理前4个重叠区域
        orig_start = block_orig_starts[i+1]  # 原始体积中的重叠区域起始
        block1_idx = i
        block2_idx = i+1
        
        # 块1的对应部分（后10层）
        block1_start = block_starts[block1_idx]
        block1_slice = blocks_3D_492[:, :, block1_start+72:block1_start+82]
        
        # 块2的对应部分（前10层）
        block2_start = block_starts[block2_idx]
        block2_slice = blocks_3D_492[:, :, block2_start:block2_start+10]
        
        # 加权混合
        for j in range(10):
            w1 = 1.0 - j/9.0
            w2 = j/9.0
            reconstructed[:, :, orig_start + j] = block1_slice[:, :, j] * w1 + block2_slice[:, :, j] * w2
    
    # 4. 处理最后一个重叠区域（20层重叠）
    # 块4的后20层和块5的前20层
    orig_start = 350
    block4_start = block_starts[4]  # 328
    block5_start = block_starts[5]  # 410
    
    block4_slice = blocks_3D_492[:, :, block4_start+62:block4_start+82]  # 块4的62-82层
    block5_slice = blocks_3D_492[:, :, block5_start:block5_start+20]  # 块5的0-20层
    
    for j in range(20):
        w1 = 1.0 - j/19.0
        w2 = j/19.0
        reconstructed[:, :, orig_start + j] = block4_slice[:, :, j] * w1 + block5_slice[:, :, j] * w2
    
    # 5. 处理中间非重叠区域
    # 块1的非重叠区域: 82-144 (来自块1的10-72层)
    reconstructed[:, :, 82:144] = blocks_3D_492[:, :, 92:154]  # 块1的10-72层
    
    # 块2的非重叠区域: 154-216 (来自块2的10-72层)
    reconstructed[:, :, 154:216] = blocks_3D_492[:, :, 174:236]  # 块2的10-72层
    
    # 块3的非重叠区域: 226-288 (来自块3的10-72层)
    reconstructed[:, :, 226:288] = blocks_3D_492[:, :, 256:318]  # 块3的10-72层
    
    # 块4的非重叠区域: 298-350 (来自块4的10-62层)
    reconstructed[:, :, 298:350] = blocks_3D_492[:, :, 338:390]  # 块4的10-62层
    
    return reconstructed

def load_data_for_worker_dcm96x192x288(base_samples: np.array, slice_start, slice_end, file_type="dcm"):
    low_pets = []
    if file_type == "dcm":
        low_pet_paths = sorted(glob(f"{base_samples}/*"))
        for low_pet_path in low_pet_paths:
            low_pet_suv = get_SUV(low_pet_path)[84: 276, 36: 324].astype(np.float32)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=2)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=0)
            low_pets.append(low_pet_suv)
    elif file_type == "npy":
        volumn = np.load(base_samples)
        num_slices = volumn.shape[0]
        for idx in range(num_slices):
            low_pet_suv = volumn[idx, :, :]
            low_pet_suv = low_pet_suv - low_pet_suv.min()
            low_pet_suv = low_pet_suv[84: 276, 36: 324].astype(np.float32)
            low_pet_suv = (low_pet_suv - low_pet_suv.min()) / (low_pet_suv.max() - low_pet_suv.min() + 1e-6)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=2)
            low_pet_suv = np.expand_dims(low_pet_suv, axis=0)
            low_pets.append(low_pet_suv)

    
            
    image_arr = np.concatenate(low_pets[slice_start: slice_end], axis=3)

    if file_type == "dcm":
        image_arr[image_arr>16] = 16
        image_arr = image_arr/16
    
    image_arr = np.transpose(image_arr, (0, 3, 1, 2))
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr = torch.from_numpy(image_arr)



    return image_arr
    


def check_shape_and_crop(arr: np.array, target_size):
    ori_arr = arr.copy()

    if len(arr.shape) > len(target_size):
        arr = np.squeeze(arr)
    assert len(arr.shape) == len(target_size), f"The input dim wrong: {arr.shape}; target shape {target_size}"

    # if arr.shape[0] != target_size[0]:
    #     raise ValueError(f"The number of layer of arr is wrong: {arr.shape}; target shape: {target_size}")
    
    if arr.shape == target_size:
        return arr
    else:
        H, W = arr.shape
        tH, tW = target_size
        
        if H < tH:
            pad_H = (tH - H) // 2
            arr = np.pad(arr, pad_width=((pad_H, pad_H), (0, 0)), mode="constant", constant_values=0)
        else:
            crop_start = H // 2 - tH // 2
            crop_end = H // 2 + tH // 2
            arr = arr[crop_start: crop_end, :]
        
        if W < tW:
            pad_W = (tW - W) // 2
            arr = np.pad(arr, pad_width=((0, 0), (pad_W, pad_W)), mode="constant", constant_values=0)
        else:
            crop_start = W // 2 - tW // 2
            crop_end = W // 2 + tW // 2
            arr = arr[:, crop_start: crop_end]

    if ori_arr.shape != arr.shape:
        pass
        # print(f"The arr heve been modify from {ori_arr.shape} to {arr.shape}")
    return arr.astype(np.float32)

if __name__ == "__main__":
    pass
    # data = load_data_for_worker_dcm432x250x250(
    #     "../Dataset/Dicom/LK/NORMAL",
    #     batch_size=1, file_type="dcm"
    # )

    # arr_result = np.zeros((192,288,492))     
    # index = 0
    # for i in range(6):
    #     batch = next(data)
    #     batch = batch["low_res"]
    #     batch = batch.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    #     ic(batch.shape)
    #     arr_result[:, :, i*82:(i+1)*82] = batch
    #     index += 1

    # final_result = postprocess(reverse_2_original_dimension_432p(arr_result))

    # import matplotlib.pyplot as plt
    # plt.imshow(final_result[:, 100, :], cmap="jet")
    # plt.savefig("temp_1.png")

    # np.save("temp", final_result)
    # ic("Fine!")
        
