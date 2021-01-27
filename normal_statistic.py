import numpy as np

frame_array = np.load('/home/lby/fabric/no_hole.npy')

seq_mean = np.mean(frame_array, axis=2)
seq_std = np.std(frame_array, axis=2)
#frame_array = (frame_array-np.expand_dims(seq_mean, axis=2))/np.expand_dims(seq_std, axis=2)
# seq_min = np.min(frame_array, axis=2)
# seq_max = np.max(frame_array, axis=2)
np.save('/home/lby/fabric/nohole_mean.npy', seq_mean)
np.save('/home/lby/fabric/nohole_std.npy', seq_std)
# np.save('/home/lby/fabric/pattern_min.npy', seq_min)
# np.save('/home/lby/fabric/pattern_max.npy', seq_max)
