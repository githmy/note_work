import copy


def get_solution(pre_condi, result):
    new_condi = copy.deepcopy(pre_condi)
    counter = 0
    while True:
        counter += 1
        print("in loop {}".format(counter))
        all_get = 1
        loop_init_condi = new_condi
        old_condi = copy.deepcopy(loop_init_condi)
        for i1 in result:
            if i1 not in old_condi:
                all_get = 0
                der_condi = derivation(old_condi)
                print("new add condition.")
                print(counter)
                new_condi = old_condi + der_condi
        if all_get == 1:
            print("solved this problem.")
        if new_condi == loop_init_condi:
            if all_get == 1:
                print("all solved!")
            else:
                print("not all solved!")
            return 0


def unfold_obj(conditions, indexobj):
    new_obj = []
    return new_obj


# 三段论 驱动 递归每一层
def derivation(conditions):
    # 1.1. 展开阶段，是否有展开的词 或 代词替换，如何按要求展开的 属性表
    # 1.2. 展开后的重命名。
    # 2.1. 逻辑属性表: 有序性，传递性
    # 2.2. 如果具有某些逻辑属性的组合，执行某种操作，如果未匹配返回警告
    for id1, i1 in enumerate(conditions):
        if 1 == 1:
            # 如果为可展开集合，则展开
            der_condi = unfold_obj(conditions, id1)
            return der_condi
    if a == 1:
        # 如果为可展开集合，则展开
        der_condi = conditions
        return der_condi


if __name__ == "__main__":
    pre_condi = []
    result = []
    get_solution(pre_condi, result)
