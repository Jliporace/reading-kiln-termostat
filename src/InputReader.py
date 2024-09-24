import cv2 
import os
import pathlib





class InputReader():
    
    def __init__(self):
        pass        

    def frame_recorded_firing(self,
                              video_path,
                              save_path):
                        
        p = sorted(pathlib.Path(video_path).glob('**/*'))
        png_compression_level = 0 
        files = [str(f) for f in p if f.is_file()]
        i = 0
        for f in files:
            cam = cv2.VideoCapture(f)
            try:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

            except OSError:
                print("Error: Creating directory of data")

            current_frame = 0
            video_fps = cam.get(cv2.CAP_PROP_FPS)

            while(True):
                cam.set(cv2.CAP_PROP_POS_MSEC,(current_frame*1000)) 
                ret, frame = cam.read()

                if (ret and current_frame < 360):
                    name = save_path + str(i) + '_' + str(current_frame) + ".png"
                    cv2.imwrite(name, frame), [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level]
                    current_frame+=60
                else:
                    break

            cam.release()
            i+=1
            # cv2.destroyAllWindows()