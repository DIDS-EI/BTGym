# 叶结点
class Leaf:
    def __init__(self, type, content, min_cost=99999, trust_cost=99999,parent_cost=99999):
        self.type = type
        self.content = content  # conditionset or action
        self.parent = None
        self.parent_index = 0
        self.min_cost = min_cost
        self.trust_cost = trust_cost
        self.parent_cost = parent_cost

    # tick 叶节点，返回返回值以及对应的条件或行动对象self.content
    def tick(self, state):
        if self.type == 'cond':
            if self.content <= state:
                return 'success', self.content
            else:
                return 'failure', self.content
        if self.type == 'act':
            if self.content.pre <= state:
                return 'running', self.content  # action
            else:
                return 'failure', self.content

    def cost_tick(self, state, cost, ticks):
        # print("self.type:", self.type, "ticks:", ticks)
        if self.type == 'cond':
            ticks += 1
            if self.content <= state:
                return 'success', self.content, cost, ticks
            else:
                return 'failure', self.content, cost, ticks
        if self.type == 'act':
            ticks += 1
            if self.content.pre <= state:
                return 'running', self.content, cost + self.content.real_cost, ticks  # action
            else:
                return 'failure', self.content, cost, ticks

    # def __str__(self):
    #     print(self.content)
    #     return ''

    def print_nodes(self):
        print(self.content)

    def count_size(self):
        return 1


# 可能包含控制结点的行为树
class ControlBT:
    def __init__(self, type):
        self.type = type
        self.children = []
        self.parent = None
        self.parent_index = 0

    def add_child(self, subtree_list):
        for subtree in subtree_list:
            self.children.append(subtree)
            subtree.parent = self
            subtree.parent_index = len(self.children) - 1

    # tick行为树，根据不同控制结点逻辑tick子结点
    def tick(self, state):
        if len(self.children) < 1:
            print("error,no child")
        if self.type == '?':  # 选择结点，即或结点
            for child in self.children:
                val, obj = child.tick(state)
                if val == 'success':
                    return val, obj
                if val == 'running':
                    return val, obj
            return 'failure', '?fails'
        if self.type == '>':  # 顺序结点，即与结点
            for child in self.children:
                val, obj = child.tick(state)
                if val == 'failure':
                    return val, obj
                if val == 'running':
                    return val, obj
            return 'success', '>success'
        if self.type == 'act':  # 行动结点
            return self.children[0].tick(state)
        if self.type == 'cond':  # 条件结点
            return self.children[0].tick(state)

    def cost_tick(self, state, cost, ticks):
        # print("self.type:",self.type,"ticks:",ticks)
        if len(self.children) < 1:
            print("error,no child")
        if self.type == '?':  # 选择结点，即或结点
            ticks += 1
            for child in self.children:
                ticks += 1
                val, obj, cost, ticks = child.cost_tick(state, cost, ticks)
                if val == 'success':
                    return val, obj, cost, ticks
                if val == 'running':
                    return val, obj, cost, ticks
            return 'failure', '?fails', cost, ticks
        if self.type == '>':  # 顺序结点，即与结点
            for child in self.children:
                # print("state:",state,"cost",cost)
                ticks += 1
                val, obj, cost, ticks = child.cost_tick(state, cost, ticks)
                if val == 'failure':
                    return val, obj, cost, ticks
                if val == 'running':
                    return val, obj, cost, ticks
            return 'success', '>success', cost, ticks
        if self.type == 'act':  # 行动结点
            return self.children[0].cost_tick(state, cost, ticks)
        if self.type == 'cond':  # 条件结点
            return self.children[0].cost_tick(state, cost, ticks)

    def getFirstChild(self):
        return self.children[0]

    # def __str__(self):
    #     print(self.type + '\n')
    #     for child in self.children:
    #         print(child)
    #     return ''

    def print_nodes(self):
        print(self.type)
        for child in self.children:
            child.print_nodes()

    # 递归统计树中结点数
    def count_size(self):
        result = 1
        for child in self.children:
            result += child.count_size()
        return result
