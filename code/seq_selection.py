from abc import ABCMeta, abstractmethod
import torch
import random
import copy
import math
import pickle
import numpy as np
import pandas as pd

def main():
    root = '/smartdata/hj7422/Documents/Workplace/Trumpf'
    threthold = 2.22
    num_epoch = 72000
    counter = 0
    model = torch.load(root + '/models/rnn_50.pkl')
    game = TS(50, threthold, model)
    agent = Agent(game, 0.6)
    for i in range(num_epoch):
        if counter % 3600 == 0:
            with open(root+'/results/seq_selection_agent.pkl', 'wb') as f:
                pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)
        agent.episode()
        counter = counter + 1

        
###################################
# self defined function and class #
###################################
class MC_node():
    def __init__(self, state, N = 0):
        self.state = state
        self.id = self.state.get_id()
        self.N = N
        self.edges = []
    def add_edge(self, e):
        self.edges.append(e)
    def add_edges(self, es):
        self.edges.extend(es)
    def is_leaf(self):
        if len(self.edges) == 0:
            return True
        else:
            return False
    def get_id(self):
        return self.id
    def get_N(self):
        return self.N
    def get_state(self):
        return self.state

class MC_edge():
    def __init__(self, action, in_node, out_node, priori):
        self.action = action
        self.in_node = in_node
        self.out_node = out_node
        self.id = in_node.get_id() + '-' + out_node.get_id()
        self.N = 0
        self.W = 0
        self.Q = 0
        self.U = priori
        self.P = priori # 获胜的概率，由网络得到
        self.value = self.Q + self.U
    def get_in_node(self):
        return self.in_node
    def get_out_node(self):
        return self.out_node
    def get_action(self):
        return self.action
    def get_state(self):
        return (self.Q, self.U, self.W, self.N, self.P)
    def get_value(self):
        # return the value of the edge, the one here is just for test --------------------
        # should add an new attribute value for the class edge???? -----------------------
        return self.value
    def set_Q(q):
        self.Q = q

class MC_tree():
    def __init__(self, root_state, cpuct, logger = None):
        self.root = MC_node(root_state)
        self.path = []
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(self.root)
    def add_node(self, node):
        if node.get_id() not in self.tree.keys():
            self.tree[node.get_id()] = node
            return 1
        else:
            print("node exist")
            return 0
    def back_fill(self, value):
        # update all the node and edge in the path
        #   - N add 1 for each edge
        #   - N add 1 for each Node 
        #   - recalculate the Q and W value of edge according to the new N value.
        for edge in self.path:
            edge.N += 1
            edge.in_node.N += 1
            edge.W = edge.W + value
            edge.Q = edge.W/edge.N
            edge.U = (self.cpuct * math.sqrt(edge.in_node.N) * edge.P)/ edge.N
            edge.value = edge.Q + edge.U
    def expansion(self, leaf, actions, states, values):
        # values here is just the output of the p_net, 
        # still not sure what kind of infos should be setted to the edge here ----------------
        # TODO
        # get the leaf node
        for action, state, value in zip(actions, states, values):
            out_node = MC_node(state)
            edge = MC_edge(action, leaf, out_node, value)
            leaf.add_edge(edge)
            self.add_node(edge.get_out_node())
    def selection(self, root, for_suggestion = False):
        # move to the leaf and save the path  
        self.path = []
        current_node = root
        #current_node = MC_node(current_state)
        if current_node.get_id() not in self.tree.keys():
            return current_node, self.path
        else:
            while not current_node.is_leaf():
                if not for_suggestion:
                    tmp_edge = max(current_node.edges, key= lambda x: x.get_value())
                else:
                    tmp_edge = max(current_node.edges, key= lambda x: x.Q)
                current_node = tmp_edge.get_out_node()
                self.path.append(tmp_edge)
            return current_node, self.path
    def set_root(self, node):
        self.root = node
        path = []

#应该区分root和current state
class Agent():
    def __init__(self, game, cpuct, logger = None):
        self.game = game
        self.root_state = game.get_current_state()
        self.mct = MC_tree(self.root_state, cpuct)
        #self.p_net = p_net
    def episode(self):
        #print('======= Selection ========')
        leaf,_ = self.mct.selection(self.mct.root)
        # test if the state of leaf staying in the done position
        #  - yes, get the reward and fill back
        #  - no, do the expansion of the leaf
        if self.game.is_done(leaf.get_state()):
            end_reward = self.game.get_reward(leaf.get_state(), True)
        else:
            #print('======= expansion ==========')
            available_actions, states, values = self.evaluate_leaf_state(leaf.get_state())
            self.mct.expansion(leaf, available_actions, states, values)
            #print('======= simulation =========')
            #end_reward = self.simulation(copy.deepcopy(leaf.get_state()))
            end_reward =self.simulation_with_random_reward(copy.deepcopy(leaf.get_state()))
        #print('======= back fill =========')
        self.mct.back_fill(end_reward)
        #print('======= end ========')
    def aborded_evaluate_leaf_state(self, state_leaf):
        """
        ABORDED!!!
        input:
          state_leaf, type of Node
        output:
          available_actions, type of list of action 
          states, type of list of State
          values, type of list of float
        """
        #state_leaf = leaf.get_state()
        available_actions = self.game.get_available_actions(state_leaf)
        states = []
        values = []
        for action in available_actions:
            new_state = self.simulate_action(copy.deepcopy(state_leaf), action)
            value_new_state = self.game.get_reward(new_state)
            states.append(new_state)
            values.append(value_new_state)
        return available_actions, states, values
    def evaluate_leaf_state(self, state_leaf):
        """
        input:
          state_leaf, type of Node
        output:
          available_actions, type of list of action
          states, type of list of State
          values, type of list of float
        """
        available_actions = self.game.get_available_actions(state_leaf)
        states = [self.simulate_action(copy.deepcopy(state_leaf), action) for action in available_actions]
        values = self.game.get_reward_batch(states)
        return available_actions, states, values
    def get_note_suggestion(self):
        tmp_edge = max(self.mct.root.edges, key= lambda x: x.Q)
        return tmp_edge.get_action()
    def get_seq_suggestion(self):
        # we can not use selection to get the suggestion because of the exporation part
        # what should we do here, here we use win rate
        # TODO 
        #???????????????????????????????
        leaf, path = self.mct.selection(self.mct.root, True)
        path = [e.get_action() for e in path]
        if self.game.is_done(leaf.get_state()):
            return path
        else:
            current_state = leaf.get_state()
            while not self.game.is_done(current_state):
                actions, states, values = self.evaluate_leaf_state(current_state)
                action, current_state, _ = max(zip(actions, states, values), key = lambda x: x[2])
                path.append(action)
            return path
    def simulate_action(self, cuu_state, action):
        return self.game.simulate_action(cuu_state, action)
    def simulation(self, tmp_state):
        while not self.game.is_done(tmp_state):
            #print('+++'+str(tmp_state.state))
            #print('++++')
            #print(tmp_state)
            actions, states, values = self.evaluate_leaf_state(tmp_state)
            #for state in states:
            #    print(state.state)
            #print(states, values)
            action, _, _ = max(zip(actions, states, values), key = lambda x: x[2])
            self.simulate_action(tmp_state, action)
        return self.game.get_reward(tmp_state, True)
    def simulation_with_random_reward(self, tmp_state):
        while not self.game.is_done(tmp_state):
            available_actions = self.game.get_available_actions(tmp_state)
            action = random.choice(available_actions)
            self.simulate_action(tmp_state, action)
        return self.game.get_reward(tmp_state, True)
    def take_action(self, action):
        # return the state after taking the action.
        self.game.take_action()
        self.root = self.game.get_current_state()
    def train_network():
        # could the episode data used to train the model again??? ---> no
        # shoud i add another model to the to predict the result and the simulation step just use random choose
        # it can enfast the process and verringt the error
        # then i can use the mcst to generate some data, and calculate the real answer. 
        # then used these data to train the model again.
        pass

class Game(metaclass = ABCMeta):
    @abstractmethod
    def get_available_actions(self, state):
        """
        input:
          state, can be any kind of type
        output:
          actions: list of action, action should be string
        """
        pass
    def get_current_state(self):
        pass
    @abstractmethod
    def is_done(self, state):
        """
        input:
          state
        output:
          out: if the game is done return the result, else return 0
        """
        pass
    @abstractmethod
    def restore_game(self):
        pass
    @abstractmethod
    def simulate_action(self):
        pass
    @abstractmethod
    def take_action(self):
        pass

class TS(Game):
    def __init__(self, lens, thread, value_network, root = '/smartdata/hj7422/Documents/Workplace/Trumpf'):
        """
        input:
          lens, type of int
              total number of parts
          thread, type of float
              value used to decide, weather the seq is good or not
          value_network, type of pkl
              neural network, that used to output the finalpunkt.
        """
        super(TS, self).__init__()
        self.polygons = pd.read_pickle(root + '/results/all_norm_features_plus_we.pkl')
        self.actions = self._load_available_actions()
        self.lens = lens
        self.thread = thread
        self.value_network = value_network   
        self.restore_game()
    def get_available_actions(self, state):
        return self.actions
    def get_current_state(self): 
        return self.current_state
    def get_original_state(self):
        return State([])
    def get_reward(self, state, is_done = False):
        # not tested jet
        inp_nn = self._transform_state_to_input(state)
        out_nn = self.value_network(inp_nn)
        score = self._transform_output_to_value(out_nn)
        # for test
        #score = np.std([float(i) for i in state.state])
        # the thread should not be constant 
        #TODO
        if not is_done:
            return score
        if score <= self.thread:
            return 1
        else:
            return -1
    def get_reward_batch(self, states):
        inp_nn = self._transform_state_to_input_batch(states)
        out_nn = self.value_network(inp_nn)
        scores = self._transform_output_to_value_batch(out_nn)
        return scores
    def _load_available_actions(self):
        self.parts = self.polygons['type'].tolist()
        self.rotations = [0, 90]
        actions = []
        num_parts = 54
        num_rotations = 2
        for i in self.parts:
            for j in self.rotations:
                actions.append((i, j))
        return actions
    def _transform_state_to_input(self, state):
        """
        input:
          state, type of State
        output:
          out, type of FloatTensor
        """
        out = transform_state_to_input(state.state, self.polygons)
        return torch.FloatTensor(out)
    def _transform_state_to_input_batch(self, states):
        """
        input:
          states, type of list of State
        output:
          out, type of FloatTensor
        """
        states = [i.state for i in states]
        out = transform_state_to_input_batch(states, self.polygons)
        return out
    def _transform_output_to_value(self, output):
        # not tested jet
        return output.mean().squeeze()
    def _transform_output_to_value_batch(self, output):
        #TODO ---------------------------------------------------------
        return output.mean(dim = 1)
    def is_done(self, state):
        if len(state.state) == self.lens:
            # done
            return True
        else:
            # not jet
            return False
    def restore_game(self):
        self.current_state = self.get_original_state()
    def simulate_action(self, cuu_state, action):
        cuu_state.take_action(action)
        return cuu_state
    def take_action(self, action):
        # change state after taking the action
        self.current_state.take_action(action)
                
class State():
    def __init__(self, state):
        self.state = state
        self.id = str(self.state)
    def __len__(self):
        return len(self.state)
    def get_id(self):
        return self.id
    def take_action(self, action):
        # action here should be type of tuple (Part, rotation)
        self.state.append(action)
        self.id = str(self.state)

class TS_rnn(torch.nn.Module):
    """
    scores for each piece
    input:
        tensor size of (batch_size, seq_len, num_dim)
    output:
        tensor size of (batch_size, seq_len)
    """
    def __init__(self, num_hidden = 64, num_layers = 2, dropout = 0.5):
        super(TS_rnn, self).__init__()
        #change the structure of the network
        num_inp = 21
        self.rnn = torch.nn.LSTM(input_size = num_inp, hidden_size = num_hidden, num_layers = num_layers, dropout = dropout)
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(num_hidden, 16),
                torch.nn.Dropout(),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1)
                )

    def forward(self, inp):
        # input of the rnn (seq_len, batch, input_size)
        data_in = torch.transpose(inp, 0, 1)
        # run rnn, it has two output
        out_rnn, _ = self.rnn(data_in)
        out_rnn = torch.transpose(out_rnn, 0, 1) # (batch_size, seq_len, num_dim)
        # rnn the mlp
        batch_size, seq_len, num_dim = out_rnn.shape
        out = []
        for i in range(seq_len):
            tmp = self.mlp(out_rnn[:, i,:])
            out.append(tmp)
        # now out is list of (batch_size, 1), combine the items in the list to get the output with size (batch_size, seq_len)
        out = torch.cat(out, 1)
        #return out.squeeze() when the batch_size == 1, this can course trouble
        return out

class TS_rnn2(torch.nn.Module):
    """
    scores only for the whole task
    input:
        tensor size of (batch_size, seq_len, num_dim)
    output:
        tensor size of (batch_size)
    """
    def __init__(self):
        super(TS_rnn2, self).__init__()
        #change the structure of the network
        num_inp = 8
        num_hidden = 64
        self.rnn = torch.nn.LSTM(input_size = num_inp, hidden_size = num_hidden, num_layers = 2)
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(num_hidden, 64),
                torch.nn.Dropout(),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
                )

    def forward(self, inp):
        # input of the rnn (seq_len, batch, input_size)
        data_in = torch.transpose(inp, 0, 1)
        # run rnn, it has two output
        out_rnn, _ = self.rnn(data_in)
        out_rnn = torch.transpose(out_rnn, 0, 1) # (batch_size, seq_len, num_dim)
        # only use the last output
        out_rnn = out_rnn[:, -1, :].squeeze()
        # rnn the mlp
        out = self.mlp(out_rnn)
        return out.squeeze()
    
class PDLoss(torch.nn.Module):
    def __init__(self, p = 2):
        super(PDLoss, self).__init__()
        self.pd = torch.nn.PairwiseDistance(p)

    def forward(self, o, t):
        # out: (batch_size, 1)
        out = self.pd(o, t)
        return out.mean()

class Data:
    """
    data class for TS_rnn
    """
    def __init__(self, x, y):
        self.data = {}
        self.data['train_x'] = self.add_file(x)
        self.data['train_y'] = self.add_file(y)[:, :, -1] # use the first metric tempately
        assert(len(self.data['train_x']) == len(self.data['train_y']))
        self.len = len(self.data['train_x'])

    def add_file(self, path):
        return torch.from_numpy(np.load(path))

    def add_scores(self, path):
        return torch.FloatTensor([float(li.rstrip('\n')) for li in open(path)])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.data['train_x'][index],
                self.data['train_y'][index])

class Data2:
    """
    data class for TS_rnn2
    """
    def __init__(self, x, y):
        self.data = {}
        self.data['train_x'] = self.add_file(x)
        self.data['train_y'] = self.add_file(y)[:, :, -1] # use the first metric tempately
        self.data['train_y'] = torch.mean(self.data['train_y'], 1)
        assert(len(self.data['train_x']) == len(self.data['train_y']))
        self.len = len(self.data['train_x'])

    def add_file(self, path):
        return torch.from_numpy(np.load(path))

    def add_scores(self, path):
        return torch.FloatTensor([float(li.rstrip('\n')) for li in open(path)])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.data['train_x'][index],
                self.data['train_y'][index])

# write the test function
def test_model(dl_test, model, loss):
    model.eval()
    test_loss = 0
    counter = 0
    for batch_idx, dat in enumerate(dl_test):
        counter += 1
        # codes to be changed
        inp, target = dat
        out = model(inp)
        lo = loss(out, target.squeeze())
        test_loss += lo.data
    return test_loss/counter

def transform_state_to_input(state, df):
    """
    FIXED STRUCTURE!!!
    input: 
      state, type of list of tuple, tuple contains: (type of part, rotation)
        sequence of inputted parts
      df, type of Dataframe, 
        inlucde all the needed features(after norm)
    output:
      out, type of torch.FloatTensor
        data, that can be used as input of the neural network.
    """
    df_tmp = []
    for i in state:
        df_tmp.append(df[df['type'] == i[0]])
    df_tmp = pd.concat(df_tmp)
    df_tmp['rotation'] = [i[1] for i in state]
    df_tmp['r1'] = df_tmp['rotation'].map(lambda x: 0 if x == 90 else 1)
    df_tmp['r2'] = df_tmp['rotation'].map(lambda x: 1 if x == 90 else 0)
    # do features selection to make sure that all the features are included and in a right sequence.
    df_tmp = df_tmp[['norm_area', 'length', 'num_of_corr', 'A/L', 
                            'area quote', 'convex area quote', 'centroid x/width',
                            'centroid y/height', 'width/height', 'convex area','area verhaltnis',
                            'r1', 'r2', 'wv0', 'wv1', 'wv2', 'wv3', 'wv4', 'wv5', 'wv6', 'wv7']]
    out = df_tmp.values.reshape(1, len(state), -1)
    return torch.from_numpy(out).float()
def transform_state_to_input_batch(states, df):
    """
    FIXED STRUCTURE!!!
    input:
        state, type of list of list of tuple, tuple contains: (type of part, rotation)
          list of sequence of inputted parts
        df, type of Dataframe,
          include all the needed fetures(after norm)
    output:
        out, type of torch.FloatTensor
          data, that can be used as input of neural network
    """
    out = []
    if len(states) == 1:
        print('i\'am signle+++++++++++')
        return transform_state_to_input(state, df)
    for state in states:
        out.append(transform_state_to_input(state, df))
    out = torch.cat(out, dim = 0)
    return out


if __name__ == '__main__':
    main()