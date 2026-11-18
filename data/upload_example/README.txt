Upload example (unseen test scene, seed 200099)
rgb_512x384.png                - RGB guidance image
depth_lowres_x4_128x96.png     - low-res depth, uint16 millimetres, 0 = missing
depth_lowres_x4_128x96.npy     - same in float32 metres, NaN = missing
depth_groundtruth_512x384.png  - ground truth, uint16 millimetres
In the Reconstruction Studio choose 'Upload low-res depth', scale 4x.
