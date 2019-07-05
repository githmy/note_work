from multiprocessing import Process
from multiprocessing import Pool
import multiprocessing
import os, time, random


# 子进程要执行的代码
def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))


def basic_process():
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')
    exit(0)


def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))


def pool_process():
    print('Parent process %s.' % os.getpid())
    cores = multiprocessing.cpu_count()
    p = Pool(2)
    for i in range(50):
        result = p.apply_async(long_time_task, args=(i,))
        if i == 3:
            break
    print('Waiting for all subprocesses done...')
    p.close()  # 会等待池中的worker进程执行结束再关闭pool
    p.join()
    p.close()
    # p.terminate()  # 则是直接关闭。
    # result.successful()  # 表示整个调用执行的状态
    print('All subprocesses done.')


if __name__ == '__main__':
    # basic_process()
    pool_process()
