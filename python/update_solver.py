import os,sys
import numpy as np
import argparse

options_to_update = {"lr_policy":"\"multistep\"", "stepvalue":[0.5,0.7], "weight_decay":"0.01", "momentum":"0.9","gamma":"0.1",
"display":0.001,"max_iter":100,"snapshot":0.01,'test_iter':1,'test_interval':0.01}

def load_solver(solver_file):
    options = []
    with open(solver_file,'r') as f:
        for line in f:
            line = line.strip()
	    if line=="":
		continue
            if line[0] == '#':
                continue
            cmd,value = [x.strip() for x in line.split(":")]
            options.append((cmd,value))        
    return options

def update_solver(options):
    options_updated = []
    for option in options:
        cmd,value = option
        if cmd in options_to_update:
            continue
        options_updated.append((cmd,value))
    return options_updated
    
def gen_solver(options, trainsize, testsize, batchsize):
    max_iter = (int)(options_to_update["max_iter"] * trainsize / batchsize)
    stepvalues = []
    for step in options_to_update["stepvalue"]:
        stepvalues.append(  int(step * max_iter ) ) 
    display = int( options_to_update['display'] * max_iter )
    snapshot = int( options_to_update['snapshot'] * max_iter )
    test_iter = int( (testsize + batchsize - 1) / batchsize )
    test_interval = int(max_iter * options_to_update['test_interval'])
    for cmd in options_to_update:
        value = options_to_update[cmd]
        if cmd == "stepvalue":
            for step in stepvalues:
                options.append( ("stepvalue", "%d"%step)  )
        elif cmd == "display":
            options.append( ("display","%d"%display)     )
        elif cmd == "snapshot":
            options.append( ("snapshot","%d"%snapshot)     )
        elif cmd == "test_iter":
            options.append( ("test_iter","%d"%test_iter))
        elif cmd == "test_interval":
            options.append( ("test_interval","%d"%test_interval))
	elif cmd == "max_iter":
	    options.append( ("max_iter", "%d"%max_iter) )
        else:
            options.append( (cmd,value ))
    return options
    
def main(solver_file,trainsize, testsize, batchsize):
    options = load_solver(solver_file)
    options = update_solver(options) 
    options = gen_solver(options, trainsize, testsize, batchsize)
    lines = []
    for option in options:
        lines.append(':'.join(option))
    lines = sorted(lines)
    lines = '\n'.join(lines)
    return lines

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-solver_file",default='solver.prototxt')
    ap.add_argument("train_size",type=int)
    ap.add_argument("test_size",type=int)
    ap.add_argument("batch_size",type=int)
    args = ap.parse_args()
    new_solver = main(args.solver_file, args.train_size, args.test_size, args.batch_size)
    output = os.path.splitext(args.solver_file)[0] + "_updated.prototxt"
    with open(output,'w') as f:
        f.write(new_solver)
    
    
            
            
