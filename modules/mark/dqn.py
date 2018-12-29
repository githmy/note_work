import numpy as np

from market_env import MarketEnv
from market_model_builder import RNN
import sys
import time
import random
from collections import deque
import simplejson
import argparse


def parseArgs(args):
    parser = argparse.ArgumentParser()
    globalArgs = parser.add_argument_group('Global options')
    globalArgs.add_argument('--modelname', default=None, nargs='?',
                            choices=["full", "one", "one_y", "one_space", "one_attent", "one_attent60"])
    globalArgs.add_argument('--learnrate', type=float, nargs='?', default=None)
    globalArgs.add_argument('--globalstep', type=int, nargs='?', default=None)
    globalArgs.add_argument('--dropout', type=float, nargs='?', default=None)
    globalArgs.add_argument('--normal', type=float, nargs='?', default=None)
    globalArgs.add_argument('--sub_fix', type=str, nargs='?', default=None)
    return parser.parse_args(args)


class bcolors:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'

    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = deque(maxlen=max_memory)
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        # if len(self.memory) > self.max_memory:
        #     del self.memory[0]
        self.memory.append([states, game_over])

    def get_batch_memory(self, batch_size=10):
        # 从经验回放池中随机取一个batch 的四元组
        batch_state = [status[0] for status, _ in random.sample(self.memory, min(len(self.memory), batch_size))]
        inputs = [i[0] for i in batch_state]
        outputs = [i[1] for i in batch_state]
        targets = [i[-1] for i in outputs]
        space_chice = [0 if i > 0 else 2 for i in targets]
        return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(space_chice,
                                                                                                 dtype=np.int32)

    def get_batch_test(self, indatas, targetdata, scope):
        # 从经验回放池中随机取一个batch 的四元组
        all_lent = targetdata.shape[0]
        inputs = []
        targets = []
        space_chice = []
        space_chice_all = [0 if i3 > 0 else 2 for i3 in targetdata]
        for i2 in range(all_lent - scope + 1):
            inputs.append(indatas[i2:i2 + scope])
            targets.append(targetdata[i2 + scope - 1])
            space_chice.append(space_chice_all[i2 + scope - 1])
        return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(space_chice,
                                                                                                 dtype=np.int32)

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        inputs = []

        # state dim
        dim = len(self.memory[0][0][0])
        for i in range(dim):
            inputs.append([])

        targets = np.zeros((min(len_memory, batch_size), num_actions))
        # 随机更新 table 每一行
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            # self.memory 第一维每条记录，第二维[[状态n-1,行为(up,down)，奖励，状态n]，结束？]
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]
            for j in range(dim):
                inputs[j].append(state_t[j][0])
            targets[i] = model.predict(state_t)[0]
            # 最大概率
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                # 记忆打折系数，根据预测值估算修正目标值
                targets[i, action_t] = reward_t + self.discount * Q_sa

        inputs = [np.array(inputs[i]) for i in range(dim)]
        return inputs, targets


def main(args=None):
    # 1. 命令行
    argspar = parseArgs(args)
    # 2. 模型参数赋值
    config = {}
    parafile = "para.json"
    if argspar.modelname is None:
        hpara = simplejson.load(open(parafile))
        config["modelname"] = hpara["model"]["modelname"]
        config["sub_fix"] = hpara["model"]["sub_fix"]
        config["tailname"] = "%s-%s" % (config["modelname"], config["sub_fix"])
    else:
        config["modelname"] = argspar.modelname
        if argspar.sub_fix is None:
            config["tailname"] = "%s-" % (argspar.modelname)
        else:
            config["sub_fix"] = argspar.sub_fix
            config["tailname"] = "%s-%s" % (argspar.modelname, argspar.sub_fix)
        parafile = "para_%s.json" % (config["tailname"])
        hpara = simplejson.load(open(parafile))
        # if argspar.sub_fix is None:
        #     config["sub_fix"] = hpara["model"]["sub_fix"]

    if argspar.learnrate is None:
        config["learn_rate"] = hpara["env"]["learn_rate"]
    else:
        config["learn_rate"] = argspar.learnrate

    if argspar.globalstep is None:
        globalstep = hpara["model"]["globalstep"]
    else:
        globalstep = argspar.globalstep
    if argspar.dropout is None:
        config["dropout"] = hpara["model"]["dropout"]
    else:
        config["dropout"] = argspar.dropout
    if argspar.normal is None:
        config["normal"] = hpara["model"]["normal"]
    else:
        config["normal"] = argspar.normal

    config["scope"] = hpara["env"]["scope"]
    config["inputdim"] = hpara["env"]["inputdim"]
    config["outspace"] = hpara["env"]["outspace"]
    config["single_num"] = hpara["env"]["single_num"]
    config["modelfile"] = hpara["model"]["file"]
    print()
    print("**********************************************************")
    print("parafile:", parafile)
    print("modelname:", config["modelname"])
    print("tailname:", config["tailname"])
    print("learn_rate:", config["learn_rate"])
    print("dropout:", config["dropout"])
    print("**********************************************************")
    print()

    # 概率小于此值时，取随机值不使用模型
    epsilon = hpara["env"]["epsilon"]
    epoch = hpara["env"]["epoch"]
    start_date = hpara["env"]["start_date"]
    end_date = hpara["env"]["end_date"]
    sudden_death = hpara["env"]["sudden_death"]
    scope = hpara["env"]["scope"]
    max_memory = hpara["env"]["max_memory"]
    batch_size = hpara["env"]["batch_size"]
    discount = hpara["env"]["discount"]

    # 3. 环境
    # env = gym.make('CartPole-v1')  # 实例化一个游戏环境，参数为游戏名称
    env = MarketEnv(input_codes=[], start_date=start_date,
                    end_date=end_date, sudden_death=sudden_death, scope=scope)

    # 4. 模型
    modelrnn = RNN(config=config)
    modelrnn.getModel()

    # 5. 记忆
    exp_replay = ExperienceReplay(max_memory=max_memory, discount=discount)

    # 6. Train
    print("Epoch0: {} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    # 每个epoch 一组固定的股票。不同epoch, 不同组股票
    for i1 in range(epoch):
        game_over = False
        # get initial input 随机抽1只数据，每个epoch固定
        try:
            input_state_next = env.reset()
            print("chara_dim is: ", env.chara_length)
            if env.done:
                continue
        except Exception as e:
            continue
        # 单支时序记忆累积
        while True:
            input_state_now = input_state_next
            input_state_next, reward, done, _ = env.step_mem()
            if done:
                break
            exp_replay.remember([input_state_now, reward, input_state_next], game_over)
            inputs, targets, space_chice = exp_replay.get_batch_memory(batch_size=batch_size)
            global_n = globalstep
            globalstep = modelrnn.batch_train(inputs, targets, space_chice, global_n)
            simplejson.dump(hpara, open(parafile, mode='w'))
        # 预测
        ori_inputs_test, ori_targets_test = env.test_state
        inputs_test, targets_test, space_chice_test = exp_replay.get_batch_test(ori_inputs_test, ori_targets_test,
                                                                                config["scope"])
        valid_list = modelrnn.valid_test(inputs_test, targets_test, space_chice_test)
        # print(valid_list)
        pred_list = modelrnn.predict(inputs_test)
        # print(pred_list)
        print("Epoch {:03d}/{} | Epsilon {:.4f}".format(i1, epoch, epsilon))
        print("Epoch: {} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


if __name__ == "__main__":
    main(sys.argv[1:])
