import numpy as np
from glob import glob 
import pydicom
import os
from pathlib import Path
from tqdm import tqdm

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


root = "asset/Anonymous_ANO_20230519_1909131_091445/D100"
dose = Path(root).stem
patient = root.split("/")[-2]
dcms = sorted(glob(f"{root}/*"))
vol = []
for dcm in tqdm(dcms):
    img = get_SUV(dcm)
    img = np.clip(img, 0, 16)
    img = np.expand_dims(img, axis=0)
    vol.append(img)
vol = np.concatenate(vol, axis=0)
os.makedirs(f"asset/whole_body/{dose}", exist_ok=True)
np.save(f"asset/whole_body/{dose}/{dose}_{patient}", vol)
