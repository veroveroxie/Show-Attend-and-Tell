import torch
import torch
import numpy as np
from PIL import Image
import os
from typing import Any, List, Tuple, Union
import sys
import random
import time


class AverageMeter(object):
    """Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, targets, k):
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def calculate_caption_lengths(word_dict, captions):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):
                continue
            else:
                lengths += 1
    return lengths


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)

def get_model_list(dirname, key, exclude='latest'):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f and exclude not in f]
    if gen_models is None or len(gen_models)<=0:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

class Saver:
    def __init__(self, model, run_dir):
        if not hasattr(model, 'model_dict'):
            raise ValueError('Please assign model_dict for this model, e.h., self.model_dict={"G":self.G}')
        self.model_dict = model.model_dict
        self.run_dir = run_dir
        self.model_dir = os.path.join(run_dir, 'model')
        self.log_dir = os.path.join(run_dir, 'log')
        self.img_dir = os.path.join(run_dir, 'img')
        make_dirs(self.model_dir)
        make_dirs(self.log_dir)
        make_dirs(self.img_dir)
        self.start_epoch, self.start_time, self.is_resume = self.resume()
        self.background_logger = Logger(os.path.join(self.log_dir, 'training_log.txt'), append=self.is_resume)
        self.logger = EasyLogger(os.path.join(self.run_dir, 'metrics.txt'), append=self.is_resume)

    @property
    def used_time(self):
        return format_time(time.time()-self.start_time)

    def resume(self):
        latest_model_name = get_model_list(self.model_dir, key='pt')
        if latest_model_name is None:
            epoch = 0
            is_resume = False
            start_time = time.time()
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            #device = 'cpu'
            model = torch.load(latest_model_name, map_location=torch.device(device))
            for key in self.model_dict:
                if self.model_dict[key] is not None:
                    self.model_dict[key].load_state_dict(model[key])
            epoch = int(os.path.basename(latest_model_name.strip(os.sep)).split('.')[0].split('-')[-1])
            is_resume = True
            start_time = time.time() - model['used_time']
            print('[*] Resuming from %s'%latest_model_name)
        return (epoch+1), start_time, is_resume

    def save_model(self, epoch):
        tmp_dict = {}
        for key in self.model_dict:
            tmp_dict[key] = self.model_dict[key].state_dict()
        tmp_dict['used_time'] = time.time() - self.start_time
        name = os.path.join(self.model_dir, 'network-snapshot-%03d.pt' % epoch)
        torch.save(tmp_dict, name)

    def save_msg(self, msg):
        self.logger.write(msg)

    def save_print_msg(self, msg):
        self.logger.write_print(msg)

    def __exit__(self):
        self.logger.close()
        self.background_logger.close()


class EasyLogger(object):
    def __init__(self, file_name, append=False):
        if append:
            file_mode = 'a'
        else:
            file_mode = 'w'
        self.file = open(file_name, file_mode)
        self.file_name = file_name
        self.file_mode = file_mode

    def write(self, message):
        self.file.write(message+'\n')
        self.file.flush()

    def write_print(self, message):
        print(message)
        self.file.write(message+'\n')
        self.file.flush()

    def close(self):
        self.file.close()

class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "a", should_flush: bool = True, append=False):
        self.file = None

        if append:
            file_mode = 'a'
        else:
            file_mode = 'w'
        self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
