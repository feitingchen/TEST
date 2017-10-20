# coding: utf-8
import numpy as np
import gym
import logging

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from gym import error, spaces, utils
from gym.utils import seeding
import math
import keras.backend as K
from keras.layers import Input, Dense, concatenate, Lambda, Conv2D, Reshape
from rl.agents.dqn import DQNAgent
from rl.core import Env
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
#read data as a pandas data frame(ACIM here)
import pandas

#這邊要放你的資料路徑
ACIM_data_frame = pandas.read_csv(filepath_or_buffer = "/home/a123456/feiting/ETF_30/ACIM.csv")

#這是我們的Data，長的超級可愛，如果你沒看過，現在讓你看看:
ACIM_data_frame
import os, sys, csv
import pandas as pd
import numpy as np
import pickle

DIRPATH = 'etf_data'

#整理data的函式
def drop(data):
    '''
    input: etf dataframe
    output: train dataframe, test dataframe, train date, test date
    '''
    data_new = data.loc[(data["Open"]!=0)&
                        (data["High"]!=0)&
                        (data["Low"]!=0)&
                        (data["Close"]!=0)&
                        (data["Volume"]!=0)&
                        (data["Adj Close"]!=0)]
    data_new = data_new.reset_index(drop=True)
    data_len_20 = int(len(data_new)/20)
    data_new_reverse = data_new.iloc[20*data_len_20::-1]
    data_new_reverse = data_new_reverse.reset_index(drop=True)
    
    train_df = data_new_reverse[0:40*20:]
    test_df  = data_new_reverse[40*20::]
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
        
    train_date = list(train_df['Date'])
    del train_df['Date']
    
    test_date = list(test_df['Date'])
    del test_df['Date']
    
    return train_df, test_df, train_date, test_date
#整理資料
ACIM_train_df, ACIM_test_df, ACIM_train_date, ACIM_train_date = drop(ACIM_data_frame)

#準備當作輸入的資料
ACIM_train_array = ACIM_train_df.values
ACIM_test_df
class ETF_Game(Env):
    def __init__(self):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low = 0, high = math.inf, shape = (20, 6))
        self.reward_range = (0., math.inf)
        
        #隨便取個名字，方便我們存資料
        self.name = "Game2048ETF"
        
        #設定隨機的seed
        self.seed()
        
        #重設遊戲
        self.reset()
        
    def reset(self):
        
        #取得資料作為遊戲盤面
        self.board = ACIM_train_array
        
        #初始化日期(由於需要觀察之前的資料作為data，故從第20天開始)
        self.cycle = 0
        self.day = 20
        
        #最大天數 ＝ 矩陣的最大長度
        self.max_day = self.board.shape[0]
        
        
        #初始現金
        self.cash = 10000
        
        #初始股票
        self.num_of_stock = 0
        self.financial_assets = 0
        
        #初始資產價值
        self.assets = self.cash+self.financial_assets
        self.assets_last_time = self.assets
        
        #初始股票價格 
        self.price = self.board[self.day-1][3]

        #表示遊戲結束與否        
        self.DONE = False        
        
        #回傳初始盤面（gym的規定）
        return self.get_observation()
        
        
    def step(self, action):
    
        #動作會介在0, 1, 2 之間，分別設為買一單位，賣一單位，不動作。(測試)
        
        # “現金” < "當前股票價格“ 時，不得購買。
        if self.price <= self.cash < 2*self.price and action == 0:
            action = 1
        if self.cash < self.price and (action == 1 or action == 0):
            action = 2
           
        # “股票數” = 0 時，不得賣出
        if self.num_of_stock == 0 and (action == 3 or action == 4):
            action = 2
        if self.num_of_stock == 1 and action == 4:
            action = 3
        
        #紀錄
        logging.debug("Action {}".format(action))
        
        #本次購買量
        purchance_quantity = 0
        
        # action
        if action == 0:
            purchance_quantity = 2
            logging.debug("Buymore")
        elif action == 1:
            purchance_quantity = 1
            logging.debug("Buy")
        elif action == 2:
            purchance_quantity = 0
            logging.debug("Hold")
        elif action == 3:
            purchance_quantity = -1
            logging.debug("Sell")
        elif action == 4:
            purchance_quantity = -2
            logging.debug("Sellmore")

         
        #已購買的股票張數更動
        self.num_of_stock += purchance_quantity
        
        #現存股票價值更動
        self.financial_assets = self.num_of_stock*self.price
        
        #現金更動
        self.cash -= purchance_quantity*self.price
        
        #總資產價值更動
        self.assets = self.cash+self.financial_assets
        
        #reward for this time
        reward = self.assets-self.assets_last_time
        
        #record today's asset
        self.assets_last_time = self.assets
        
        #週期數+1
        self.cycle += 1
        self.day += 20

        #update股票價格 
        self.price = self.board[self.day-1][3]
        
        #get now observation for today
        observation = self.get_observation()
        
        #當自身資產歸零，或是到達最大天數時，遊戲結束，其餘均繼續進行。
        done = None
        if self.assets == 0 or self.day == self.max_day:
            done = True
            self.DONE = True
        else:
            done = False
        
        #這裡可以回傳特定資料作為紀錄（格式是字典檔），因為我們目前沒有需要，所以隨便設個空的字典檔。
        info = dict()
        
        #這裡要回傳什麼，回傳的順序，都是gym規定的
       
        return observation, reward, done, info
            
            
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_observation(self):
        #get board for last 20 days
        return self.board[self.day-20:self.day]

    #這裡是拿來做test時候的顯示        
    def render(self, mode='human', close=False):
        if close:
            return
        outfile = None
        if self.DONE == True:
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            s = "total assets:" + str(self.assets) + "\n"
            s += "cash left" + str(self.cash) + "\n"
            s += "num of stock left" + str(self.num_of_stock) + "\n"
            s += "final price of ETF" + str(self.price) + "\n"
            s += "financial_assets left" + str(self.financial_assets) + "\n"
            outfile.write(s)
        return outfile
#把我們辛苦架好的遊戲環境作為測試環境
ETF_env = ETF_Game()
nb_actions = ETF_env.action_space.n

#可以執行的動作數
nb_actions
BOARD_INPUT_SHAPE = (20, 6)
WINDOW_LENGTH = 1

#這裡的window_length 是指當我需要傳入包括前幾次畫面作為資料時的東西，
#他是把它當作CNN的channel數一樣的東西
#本來這裡是不需要加的，只是keras-rl寫死了所以我只好傳進去。

#另外，由於資料最後一步是keras-rl處理的，他的順序這樣寫，
#我也只好這樣寫
input_shape = (WINDOW_LENGTH,) + BOARD_INPUT_SHAPE 

#設定輸入層的形狀
model_input = Input(shape = input_shape)

#視不同的backend要排一下順序
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    permute = Permute((2, 3, 1), input_shape=input_shape)
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    permute = Permute((1, 2, 3), input_shape=input_shape)
    
#把排列的結果套用上去，喬一下我們的原始input
preprocessed_input = permute(model_input)

#橫著看～
conv_horizontal_1 =  Conv2D(filters = 32, kernel_size = (1, 6), padding='valid', activation = "relu")

#直的看～
conv_vertical_1 = Conv2D(filters = 32, kernel_size = (4, 1), padding='valid', activation = "relu")

#把處理過的input塞進去
layer_horizontal_1 = conv_horizontal_1(preprocessed_input)
layer_vertical_1 = conv_vertical_1(preprocessed_input)

#再橫著看
conv_horizontal_2 =  Conv2D(filters = 64, kernel_size = (1, 6), padding='valid', activation = "relu")

#在直著看
conv_vertical_2 = Conv2D(filters = 64, kernel_size = (4, 1), padding='valid', activation = "relu")

#交錯塞
layer_h_then_v_2 = conv_vertical_2(layer_horizontal_1)
layer_v_then_h_2 = conv_horizontal_2(layer_vertical_1)

#把上面兩個拉直
flat_h_then_v = Flatten()(layer_h_then_v_2)
flat_v_then_h = Flatten()(layer_v_then_h_2)

#接在一起～


conv_merge = concatenate([flat_h_then_v, flat_v_then_h], name = "Merge_Layer")

action = Dense(5)

output_action = action(conv_merge)

model = Model(model_input, output_action)
model.summary()
#來測試一下：

#隨便產生一組資料
test_board = np.random.random_sample(size = (50, 1, 20, 6))
test_board *= 60

#送進去跑看看～
model.predict(test_board)


mode = input("Mode?")
step = int(input("Step?"))

#設定記憶體
memory = SequentialMemory(limit=10000, window_length=1)

#set policy
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=step)

#DQN設定
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory = memory, policy = policy,
               nb_steps_warmup=100000, gamma=.90, target_model_update=50,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.0001), metrics=['mae'])
if mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(ETF_env.name)
    checkpoint_weights_filename = 'dqn_' + ETF_env.name + '_weights2_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(ETF_env.name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=100000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    
    #load weights
    #dqn.load_weights("dqn_Game2048Env_weights2_10000000.h5f")
    
    dqn.fit(ETF_env, callbacks=callbacks, nb_steps=step, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    #dqn.test(env, nb_episodes=10, visualize=False)
    
elif mode == 'test':
    weights = "dqn_"+ETF_env.name+"_weights2_10000000.h5f"
    print("weights: ", weights)
    if weights:
        weights_filename = weights
    dqn.load_weights(weights_filename)
    dqn.test(ETF_env, nb_episodes=10, visualize=True)
get_ipython().magic('save DQN_ETF_30DAY 1-14')
