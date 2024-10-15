import cv2 
import os
import pathlib


PNG_COMPRESSION_LEVEL = 0 

class InputReader():
    #TO-DO refactor frame_firing, save_cropped_images from CurveCreator to InputReader
    def __init__(self):
        pass        

    def frame_recorded_firing(self,
                              video_path: str,
                              save_path: str):
        """
        For each video in video path, frame video with 1 minute interval. Save frames to save_path
        
        """
                        
        p = sorted(pathlib.Path(video_path).glob('**/*'))
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
                    cv2.imwrite(name, frame), [int(cv2.IMWRITE_PNG_COMPRESSION), PNG_COMPRESSION_LEVEL]
                    current_frame+=60
                else:
                    break

            cam.release()
            i+=1
            # cv2.destroyAllWindows()