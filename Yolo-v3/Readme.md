# Yolo-v3
## Training
1.Generate your own annotation file and class names file.  
One row for one image;  
Row format: image_file_path box1 box2 ... boxN;  
Box format: x_min,y_min,x_max,y_max,class_id (no space).   
For VOC dataset, try `python voc_annotation.py`    
2.Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`    
The file model_data/yolo_weights.h5 is used to load pretrained weights.  
3.Modify train.py and start training.  
`python train.py`  
Use your trained weights or checkpoint weights with command line option --model model_file when using yolo_video.py Remember to modify class path or anchor path, with --classes class_file and --anchors anchor_file.  

## Testing
To test the model on image, run the following code  
`python yolo_video.py [OPTIONS...] --image, for image detection mode, OR`  
`python yolo_video.py [video_path] [output_path (optional)]`  
