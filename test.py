from dataset import CustomDataset
from hparams import hparams
import os
from train import create_model
data_dir='./ljspeeh'
from torch.utils.data import DataLoader
import torch
import numpy as np
model  = create_model(hparams)

dataset = CustomDataset(meta_file=os.path.join(data_dir, 'train.txt'),
                            receptive_field=model.receptive_field,
                            sample_size=hparams.sample_size,
                            upsample_factor=200,
                            quantization_channels=hparams.quantization_channels,
                            use_local_condition=True,
                            upsample_network=False,
                            noise_injecting=hparams.noise_injecting,
                            feat_transform=None)

dataloader = DataLoader(dataset, batch_size=hparams.batch_size,
                             shuffle=True, num_workers=2,
                             pin_memory=True)

for i in range(len(dataset)):
    x,y,z=dataset[i]
    if  z.shape[0]!=18048:
        print(x.shape)
        print(y.shape)
        print(z.size())

def test_upsample_network(z):
    from fftnet_gaussian import UpSampleConv
    upsample_conv = UpSampleConv()
    z = z.view(1,*z.size())
    z = z.unsqueeze(1)
    z = upsample_conv(z)
    z = z.squeeze(1)
    return z

def test_prepare(condition_mel_file,audio_file):
    condition = np.load(condition_mel_file)
    audios = np.load(audio_file)
    condition = torch.from_numpy(condition.transpose(1,0)).float()
    condition = test_upsample_network(condition)
    assert condition.size(-1)==audios.shape[0]


if __name__ == '__main__':

    audio_path='/home/jinqiangzeng/work/pycharm/FFTNet/training_data/audios/arctic_a0001.npy'
    mel_path='/home/jinqiangzeng/work/pycharm/FFTNet/training_data/mels/arctic_a0001.npy'
    test_prepare(mel_path,audio_path)