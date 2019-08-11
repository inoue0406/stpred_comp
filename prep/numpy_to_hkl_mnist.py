import numpy as np
import hickle as hkl
import matplotlib as plt
from sklearn import preprocessing as p
mnist_size=10000
#LENGTH_OF_VID =20*4000#20*50#20*10000##50#20*10000#20*mnist_size # For later experiments, modify size as necessary20*1000
IM_SZ_WID = 64
IM_SZ_HGT=64
VIDEO_CHANNELS = 1

dname = '../data/mnist/'
train_video=np.load(dname+'mnist_test_seq.npy')

def video_generator(istart,iend):
    LENGTH_OF_VID =20*(iend-istart)
    in_video = np.empty([LENGTH_OF_VID, IM_SZ_HGT, IM_SZ_WID, VIDEO_CHANNELS], dtype=np.float32)#LENGTH_OF_VID was 1st attribute

    count =0
    for i in range (istart, iend):
           gray_frame= train_video[:,i,:,:]
           gray_frame = np.expand_dims(gray_frame, axis=3)
           in_video[0+count:count+20,:,:,:]= np.copy(gray_frame)
           count=count+20
    return in_video

def save_as_hickle(istart,iend,dname,fname):
    in_video = video_generator(istart,iend)
    num_frames = in_video.shape[0]
    print(in_video.shape)
    source_string = ["mnist"]*num_frames
    # dump data to file
    print( in_video.shape)
    hkl.dump(in_video, dname+fname+'_data.hkl', mode='w')
    # dump names to file
    import pdb; pdb.set_trace()
    hkl.dump(source_string, dname+fname+'_sources.hkl', mode='w')

save_as_hickle(0,6000,dname,'mnist_tain_6000')
save_as_hickle(6000,8000,dname,'mnist_valid_2000')
save_as_hickle(8000,10000,dname,'mnist_test_2000')


