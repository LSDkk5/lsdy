#!/usr/bin/python
from numpy import random
from time import time
from os import path
import argparse
import os

class LsdY:
    def __init__(self):
        self.__workPath = os.getcwd()
        self.labelPath = os.path.join(self.__workPath, 'data/label.names')
        self.weightsPath = os.path.join(self.__workPath, 'weights/416.weights')
        self.configsPath = os.path.join(self.__workPath, 'cfg/416.cfg')

        self.inputPath = os.path.join(self.__workPath, 'data/input')
        self.outputPath = os.path.join(self.__workPath, 'data/output')

        self.CONFIDENCE = 0.5
        self.TRESHOLD = 0.4

        self.scriptStart = time()

    def load_labels(self):
        LABELS = open(self.labelPath).read().strip().split('\n')
        return LABELS

    def set_colors(self, seed):
        random.seed(seed)
        l = self.load_labels()
        COLORS = random.randint(0 , 255, size=(len(l), 3), dtype='uint8')
        return COLORS

    def run(self):
        ap = argparse.ArgumentParser()
        atype = ap.add_mutually_exclusive_group(required=True)
        atype.add_argument("-i", "--image", action="store_true")
        atype.add_argument("-v", "--video", action="store_true")
        atype.add_argument("-l", "--live", action="store_true")
        ap.add_argument("--preview", required=False, action="store_true")
        ap.add_argument("--showp", required=False, action="store_true",
                        help="show percentage")
        ap.add_argument("--config", required=False)
        ap.add_argument("--weights", required=False)
        ap.add_argument("-c", "--confidence", type=float, required=False)
        ap.add_argument("-t", "--treshold", type=float, required=False)
        ap.add_argument("-in", "--input", required=False)
        ap.add_argument("-ou", "--output", required=False)
        args, __ = ap.parse_known_args()


        if args.config!=None:
            self.__configPath = args.cfg
        if args.weights!=None:
            self.__weightPath = args.weights

        try:
            if args.confidence!=None:
                if args.confidence<1 and args.confidence>0:
                    self.CONFIDENCE = args.confidence
                else: raise ConfidenceError
            if args.treshold!=None:
                if args.treshold<1 and args.treshold>0:
                    self.TRESHOLD = args.treshold
                else: raise TresholdError
        except TresholdError:
            print(f"\n{path.basename(__file__)}: error: argument -c/--confidence: must be beetwen 0 and 1")
            exit()
        except ConfidenceError:
            print(f"\n{path.basename(__file__)}: error: argument -t/--treshold: must be beetwen 0 and 1")
            exit()
            
        if args.input!=None:
            self.__inputPath = args.input
        if args.output!=None:
            self.__outputPath = args.output
        
        try:
            from d_video import DetectObjectsFromVideo
            from d_image import DetectObjectsFromImage
            if args.showp:
                if args.image:
                    DetectObjectsFromImage().start(True)
                elif args.video:
                    DetectObjectsFromVideo().start(False, True)
                elif args.live:
                    DetectObjectsFromVideo().start(True, True)
            else:
                if args.image:
                    DetectObjectsFromImage().start(False)
                elif args.video:
                    DetectObjectsFromVideo().start(False, False)
                elif args.live:
                    DetectObjectsFromVideo().start(True, False)
        except Exception as e:
            print(e)

class ConfidenceError(Exception):
    pass
class TresholdError(Exception):
    pass

if __name__ == "__main__":
    st = LsdY().run()