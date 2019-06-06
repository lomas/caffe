#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time
from tqdm import tqdm
import caffe

def load_list(filepath,root):
    samples = []
    try:
        with open(filepath,'r') as f:
            for line in tqdm(f):
                path,cid = line.strip().split(' ')
                if root != "":
                    path = os.path.join(root,path)
                samples.append((path,cid))
    except Exception,e:
        print ('oops! ',e)
    return samples
            
def convert_score_to_string(preds):
    results = []
    for pred in preds:
        pred = map(lambda x: "%.3f"%x, pred.tolist())
        results.append(' '.join(pred))
    return results
        
    
def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or txt."
    )
    parser.add_argument(
        "output_file",
        help="Output txt filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        help="Trained model weights file."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="-1 for cpu 0 for gpu 0 and so on."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default="",
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--mean_bgr",
        default="0,0,0",
        help="mean value for bgr channels"
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=1.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size for predicting"
    )
    parser.add_argument(
        "--input_root",
        default="",
        help="root folder for path in list file"
    )    
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    elif args.mean_bgr:
        mb, mg, mr = args.mean_bgr.split(',')
        mean = np.zeros( [3] + [int(x) for x in args.images_dim.split(',')] )
        mean[0,:,:] = float(mb)
        mean[1,:,:] = float(mg)
        mean[2,:,:] = float(mr)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.device >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.device)
        print("GPU: ",args.device)
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.txt), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('txt'):
        print("Loading file: %s" % args.input_file)
        inputs = load_list(args.input_file,args.input_root)
    elif os.path.isdir(args.input_file):
        print("Loading folder: %s" % args.input_file)
        inputs =[im_f for im_f in glob.glob(args.input_file + '/*.' + ".jpg")]
    else:
        print("Loading file: %s" % args.input_file)
        #inputs = [caffe.io.load_image(args.input_file)]
        inputs = [args.input_file]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    preds = []
    aligned_num = len(inputs)//args.batch_size * args.batch_size
    for k_ind in tqdm(range(0, aligned_num , args.batch_size)): 
        imgs =[ caffe.io.load_image(im_f[0]) for im_f in inputs[k_ind:k_ind+args.batch_size]]
        predictions = classifier.predict(imgs, not args.center_only)
        preds.extend( convert_score_to_string(predictions) )
    if aligned_num < len(inputs):
        imgs =[ caffe.io.load_image(im_f[0]) for im_f in inputs[aligned_num:]]
        predictions = classifier.predict(imgs, not args.center_only)
        preds.extend( convert_score_to_string(predictions) )
    print("Done in %.2f s." % (time.time() - start))
    
    
    # Save
    print("Saving results into %s" % args.output_file)
    lines = map(lambda (a,b): ' '.join( [a[0],a[1],b] ), zip(inputs,preds))
    with open(args.output_file,'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv)
