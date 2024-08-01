import sys
sys.path.append('/home/jessica/reading-kiln-termostat/src')
import InputReader
import TesseractPipeline


reader = InputReader.InputReader()
tesseract_pipeline = TesseractPipeline.TesseractPipeline()
video_path = "/home/jessica/reading-kiln-termostat/data/gisela"
save_path = "./kiln-images/color-images/"


reader.read_video(video_path,
                 save_path,
                 500)