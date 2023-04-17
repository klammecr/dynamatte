# dynamatte
Dynamatte: A dynamic matting method to generate in scene video fusion

# Baselines:
- [x] Validate dataset to be used in initial work
## Extraction
- [x] Extract out omnimattes
- [ ] Omnimatte extraction for custom videos
- [ ] Extraction differences from layered neural atlases
## Blending
- [x] Existing omnimatte data for testing
- [ ] Utilize results from extraction
- [ ] Lazy "cut and paste" of extract omnimattes
- [ ] Poisson image blending
- [ ] Blending from background matting paper
- [ ] STRETCH: Use GP-GAN for blending

*Richa
+Achleschwar
$Chris

## Pulling down the repo with omnimatte
```
git pull --recurse-submodules
git submodule update --init --recursive --remote
```

## Running Video Preprocessing
```
python video_preprocessing/process_video.py -v video_preprocessing/sailboats.mp4 -o datasets/sailboat
```

## Visualization of Homography
```
python video_processing/visualize_homography -v datasets/tennis/rgb -hf datasets/tennis/homographies.txt
```