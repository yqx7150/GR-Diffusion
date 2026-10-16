import numpy as np
import pydicom
from glob import glob
import torch
from tqdm import tqdm
import os

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = f"/home/lqg/code/czj/data/dicom/Anonymous_ANO_20230626_1601550_092415/2.886 x 600 WB NORMAL"
pets = sorted(glob(f"{root}/*"))
vol = []
for pet in pets:
    suv = get_SUV(pet)
    suv = np.clip(suv, a_min=0, a_max=16)
    suv = suv / 16
    suv = np.expand_dims(suv, axis=0)
    vol.append(suv)
vol = np.concatenate(vol, axis=0)

save_path = f"./asset/whole_body/Anonymous_ANO_20230626_1601550_092415/NORMAL"
os.makedirs(save_path, exist_ok=True)

C, H, W = vol.shape
shape_str = f"{C}x{H}x{W}"

np.save(f"{save_path}/Anonymous_ANO_20230626_1601550_092415_{shape_str}_.npy", vol)


