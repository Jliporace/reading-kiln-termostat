import cv2 
import os
import pathlib

class InputReader():
    
    def __init__(self):
        pass

    def read_video(self,
                   video_path,
                   save_path,
                   saving_rate = 2):
        
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

            while(True):
                ret, frame = cam.read()

                if ret:
                    if(current_frame % saving_rate == 0):
                        name = save_path + str(i) + '_' + str(current_frame) + ".jpg"
        #                 print("Creating..." + name)
                        cv2.imwrite(name, frame)
                        
                    current_frame+=1
                else:
                    break

            cam.release()
            i+=1
            # cv2.destroyAllWindows()