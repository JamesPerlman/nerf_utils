import os
import subprocess as sp
import sys

# convenience method to fetch gpu indices via `nvidia-smi`
def get_gpus(args: dict) -> list[str]:
    proc = sp.Popen(['nvidia-smi', '--list-gpus'], stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = proc.communicate()
    data = [line.decode() for line in out.splitlines(False)]
    gpus = [f"{item[4:item.index(':')]}" for item in data]
    if args.gpus and args.gpus != "all":
        gpus = [id for id in gpus if id in args.gpus]

    return gpus

# this splits the work into subprocesses, one for each gpu
# args must have a --batch flag, which is used to determine which gpu to use
# args must have a --gpus flag, which is used to determine which gpus to use
# it will run the calling script again, but with a --batch flag per gpu
def split_across_gpus(args, n_elements, run_fn, final_fn=None):
    if args.batch != None:
        print("Starting process on CUDA device: " + os.environ['CUDA_VISIBLE_DEVICES'])

        [proc_idx, n_procs] = [int(s) for s in args.batch.split('/')]
        run_fn(proc_idx, n_procs)
        
    # No --batch flag means we are part of the main process
    else:

        # split into subprocesses, one for each gpu
        procs = []
        gpus = get_gpus(args)
        n_gpus = len(gpus)

        # In case there are less elements to process than the number of gpus available...
        if n_elements < n_gpus:
            gpus = gpus[0:n_elements]
            n_gpus = n_elements

        print(f"Using {n_gpus} GPU(s).  Processing...")
        
        i = 0
        for gpu in gpus:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            
            # rerun this command, but with a batch arg
            cmd = sys.argv.copy()
            cmd.insert(0, 'python')
            cmd.extend(["--batch", f"{i}/{n_gpus}"])

            proc = sp.Popen(cmd, env=env, shell=True, stderr=sys.stderr, stdout=sys.stdout)
            procs.append(proc)

            i = i + 1
        
        for p in procs:
            p.wait()
        
        if final_fn:
            final_fn()
