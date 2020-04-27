#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils
import numpy as np

import argparse
import sys
import cv2

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in an image using an object detection DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("file_in", type=str, help="filename of the input video to process")
parser.add_argument("file_out", type=str, default=None, nargs='?', help="filename of the output image to save")
parser.add_argument("--network", type=str, default="pednet", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="none", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
parser.add_argument("--device", type=str, default="DLA", help="Device to use. Either GPU or DLA")
parser.add_argument("--precision", type=str, default="FP16", help="Either INT8, FP16, FP32")

try:
	opt = parser.parse_known_args()[0]
	print(opt)
except:
	print("")
	parser.print_help()
	sys.exit(0)

video = cv2.VideoCapture(opt.file_in)

if not video.isOpened():
	print("Could not open video")
	sys.exit()

class JetInf():
	def __init__(self, opt):
		self.net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
		self.opt = opt
	def detect(self, cuda_mem, width, height):
		detections = self.net.Detect(cuda_mem, width, height, self.opt.overlay)
		return self.nms(detections)
	def nms(self, detections, overlapThresh=0.3):
		if len(detections) == 0: return detections
		order = np.array([i[0] for i in sorted(enumerate(detections), key=lambda detection: detection[1].Confidence, reverse=True)])
		keep = []
		while order.size > 0:
			i = order[0]
			keep.append(i)
			xx1 = np.maximum(detections[i].Left, detections[order[1:]].Left)
			yy1 = np.maximum(detections[i].Bottom, detections[order[1:]].Bottom)
			xx2 = np.minimum(detections[i].Right, detections[order[1:]].Right)
			yy2 = np.minimum(detections[i].Top, detections[order[1:]].Top)
			w = np.maximum(0.0, xx2 - xx1 + 1)
			h = np.maximum(0.0, yy2 - yy1 + 1)
			inter = w * h
			ovr = inter / (detections[i].Area + detections[order[1:]].Area - inter)
			inds = np.where(ovr <= overlapThresh)[0]
			order = order[inds + 1]
		return detections[keep]

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

jetinf = JetInf(opt)
while True:
	ok, frame = video.read()
	if not ok: break
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
	cuda_mem = jetson.utils.cudaFromNumpy(frame)

	detections = jetinf.detect(cuda_mem, width, height)
	print("detected {:d} objects in image".format(len(detections)))
	#cv2.imshow("frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): break

net.PrintProfilerTimes()

