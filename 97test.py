import tensorflow as tf
import horovod.tensorflow as hvd

# 1. 初始化 Horovod
hvd.init()

# 2. 一个 GPU 与一个进程绑定
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

# Build model...
loss = ...

# 3. 根据总 GPU 数量放大学习率
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())
# 4. 使用 hvd.DistributedOptimizer 封装原有的 optimizer。只是梯度同步由 hvd.DistributedOptimizer 负责。
opt = hvd.DistributedOptimizer(opt)

# Add hook to broadcast variables from rank 0 to all other processes during
# 5. 广播初始变量值到所有进程
hooks = [hvd.BroadcastGlobalVariablesHook(0)]

# Make training operation
train_op = opt.minimize(loss)

# 6. 只在 worker 0 上保存 checkpoint
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, config=config, hooks=hooks) as mon_sess:
    while not mon_sess.should_stop():
        # Perform synchronous training.
        mon_sess.run(train_op)

# 在单机 4 卡的机上起训练，只需执行以下命令：
# horovodrun -np 4 -H localhost:4 python train.py
# 在 4 机，每机 4 卡的机子上起训练，只需在一个机子上执行以下命令即可：
# horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py
