import socket
import os


def pin_workers_iterator(the_iterator, args):
    if 'karolina' in socket.gethostname() or 'ristoteles' in socket.gethostname() or os.path.exists("/project/project_465000165"):
        return
    try:
        print(args.cpus)
    except AttributeError:
        args.cpus = list(sorted(os.sched_getaffinity(0)))

    if args.num_workers > 0:
        for index, w in enumerate(the_iterator._workers):
            os.system("taskset -p -c %d %d" % ((args.cpus[(index + 1) % len(args.cpus)]), w.pid))
