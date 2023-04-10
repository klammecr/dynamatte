# dynamatte
Dynamatte: A dynamic matting method to generate in scene video fusion

# Baselines:
- [ ] *+$ Validate dataset to be used in initial work
## Extraction
- [ ] + Extract out omnimattes
- [ ] Extraction differences from layered neural atlases
- [ ] D2-NeRF
## Blending
- [ ] $ Existing omnimatte data for testing
- [ ] Utilize results from extraction
- [ ] Lazy "cut and paste" of extract omnimattes
- [ ] Poisson image blending
- [ ] Blending from background matting paper
- [ ] STRETCH: Use GP-GAN for blending

*Richa
+Achleschwar
$Chris


## Running Video Preprocessing
```
python video_preprocessing/process_video.py -v video_preprocessing/sailboats.mp4 -o datasets/sailboat
```
