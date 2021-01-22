import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pygame
from pygame.locals import *
import gym
from os import path,listdir
from tqdm import tqdm
from PIL import Image, ImageDraw
import pickle



DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
NUMX = NUMY = 20
RED = (255,64,64)
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (127,127,127)
SANDBOX = True
POLE_LENGTH = 50
ARROW_BASE_SIZE = 7
CURRENT_DIR = path.abspath(path.curdir)


def calc_exit_time(observation):
    
    return (np.sign(observation[1])*2.4 - observation[0])/observation[1]

def calc_fall_time(observation):
    
    return (np.sign(observation[3])*0.066*math.pi - observation[2])/observation[3]



class MarkovAgent():
    
    def __init__(self):
        
        self.memory = []
        self.Qs = {}
        self.Ns = {}
        self.done = False
        self.gamma = 0.9
        self.epsilon = 1
        self.decay = 0.99
        
        self.session_score = 0
        self.max_distance = 0
        
        self.env = gym.make("CartPole-v3")
        
    def reset(self):
        
        self.done = False
        self.memory = []
        
    def play_session(self):
        
        observation = self.env.reset()
        
        self.session_score = 0
        self.max_distance = 0
        
        while not self.done:
            
            s,action = self.take_action(observation)
            observation, reward, self.done, info = self.env.step(action)
            self.session_score += 1            
            self.memory.append([s,-100 if self.done and self.session_score<2000 else reward,action])
            
            distance = observation[0]
            if abs(distance)>self.max_distance:
                self.max_distance = abs(distance)
        

            
        past_reward = 0
        
        for s, reward, action in reversed(self.memory):
            
            self.Qs[s][action] = (self.Qs[s][action]*self.Ns[s][action] + (reward + self.gamma*past_reward))/(self.Ns[s][action]+1)
            self.Ns[s][action] += 1
            past_reward = (reward + self.gamma*past_reward)
            
        self.epsilon = max(self.decay*self.epsilon,0.05)
        self.reset()
        
        
    def take_action(self,observation):
        
        s = np.asarray([round(observation[i],1+int(i/2)) for i in range(len(observation))]).tostring()
        
        if s not in self.Ns:
            
            self.Ns[s] = [5,5]
            self.Qs[s] = [100,100]

        q_list = self.Qs[s]

        
        if np.random.uniform(0,1)>self.epsilon:
            
            action = np.argmax(q_list)
            
        else:
            
            action = self.env.action_space.sample()
            
        return s,action
            
        
        




if not SANDBOX:

	env = gym.make("CartPole-v1")
	observation = env.reset()
	done = False

	# model = pickle.load(open("models/svm1.pkl","rb"))
	agent = pickle.load(open("models/mark.pkl","rb"))



x,y = np.meshgrid(np.linspace(-0.2,0.2,20),np.linspace(-1.5,1.5,20))

speedx = x.copy()

for i in range(20):
    for j in range(20):
        
        speedx[i,j] = math.sin(speedx[i,j])

u = y
v = speedx

mousex,mousey = 204,302

import pylab

fig = pylab.figure(figsize=[6, 6], # Inches
                   dpi=50,        # 50 dots per inch, so the resulting buffer is 300x300 pixels
                   )
ax = fig.gca()
ax.quiver(x,y,u,v,scale=40)
ax.set_title("Phase Space Diagram")
ax.set_ylabel("Angular Velocity")
ax.set_xlabel("Angle")

canvas = agg.FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()


pygame.init()

window = pygame.display.set_mode((DISPLAY_WIDTH,DISPLAY_HEIGHT))
screen = pygame.display.get_surface()

size = canvas.get_width_height()

surf = pygame.image.fromstring(raw_data, size, "RGB")
screen.blit(surf, (50,150))
pygame.display.flip()


i = 0
crashed = False

while not crashed:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			crashed = True

		if SANDBOX:

			if pygame.mouse.get_pressed()[0]:

				coords = pygame.mouse.get_pos()

				if 89<coords[0]<319 and 187<coords[1]<416:

					mousex,mousey = coords



			theta = (mousex-204)*2/(210*5)
			omega = -3*(mousey-302)/(210)

		else:

			theta = observation[2]
			omega = observation[3]



		ending_pointx = 600 + POLE_LENGTH*math.sin(theta)
		ending_pointy = 300 - POLE_LENGTH*math.cos(theta)

		if omega<0:
			arrow = True
			arrow_anchor = (600 + 1.5*POLE_LENGTH*math.sin(theta),300 - 1.5*POLE_LENGTH*math.cos(theta))
			arrow1 = (arrow_anchor[0] - omega*ARROW_BASE_SIZE*math.sin(theta),arrow_anchor[1] - -1*omega*ARROW_BASE_SIZE*math.cos(theta))
			arrow2 = (arrow_anchor[0] - omega*ARROW_BASE_SIZE*math.cos(math.pi/2-theta) - 2*omega*ARROW_BASE_SIZE*math.cos(theta),arrow1[1] + -1*3*omega*ARROW_BASE_SIZE*math.sin(theta))
			arrow3 = (arrow2[0] - 2*omega*ARROW_BASE_SIZE*math.sin(theta),arrow2[1] - -1*2*omega*ARROW_BASE_SIZE*math.cos(theta))
			arrow4 = (arrow1[0] - 2*omega*ARROW_BASE_SIZE*math.sin(theta),arrow1[1] - -1*2*omega*ARROW_BASE_SIZE*math.cos(theta))
			arrow5 = (arrow4[0] - omega*ARROW_BASE_SIZE*math.sin(theta),arrow4[1] - -1*omega*ARROW_BASE_SIZE*math.cos(theta))
			arrow6 = ((arrow1[0]+arrow4[0])/2 + 1.5*omega*ARROW_BASE_SIZE*math.cos(theta),(arrow1[1]+arrow4[1])/2 - -1*1.5*omega*ARROW_BASE_SIZE*math.sin(theta))

		elif omega>0:
			arrow = True
			arrow_anchor = (600 + 1.5*POLE_LENGTH*math.sin(theta),300 - 1.5*POLE_LENGTH*math.cos(theta))
			arrow1 = (arrow_anchor[0] + omega*ARROW_BASE_SIZE*math.sin(theta),arrow_anchor[1] - omega*ARROW_BASE_SIZE*math.cos(theta))
			arrow2 = (arrow_anchor[0] - omega*ARROW_BASE_SIZE*math.cos(math.pi/2-theta) - 2*omega*ARROW_BASE_SIZE*math.cos(theta),arrow1[1] - 2*omega*ARROW_BASE_SIZE*math.sin(theta))
			arrow3 = (arrow2[0] + 2*omega*ARROW_BASE_SIZE*math.sin(theta),arrow2[1] - 2*omega*ARROW_BASE_SIZE*math.cos(theta))
			arrow4 = (arrow1[0] + 2*omega*ARROW_BASE_SIZE*math.sin(theta),arrow1[1] - 2*omega*ARROW_BASE_SIZE*math.cos(theta))
			arrow5 = (arrow4[0] + omega*ARROW_BASE_SIZE*math.sin(theta),arrow4[1] - omega*ARROW_BASE_SIZE*math.cos(theta))
			arrow6 = ((arrow1[0]+arrow4[0])/2 + 1.5*omega*ARROW_BASE_SIZE*math.cos(theta),(arrow1[1]+arrow4[1])/2 + 1.5*omega*ARROW_BASE_SIZE*math.sin(theta))

		else:
			arrow = False




		screen.fill((BLACK))
		pygame.draw.rect(screen,GRAY,(5,5,DISPLAY_WIDTH-10,DISPLAY_HEIGHT-10))
		pygame.draw.rect(screen,BLACK,(47,147,306,306))
		pygame.draw.rect(screen,BLACK,(447,147,306,306))
		screen.blit(surf, (50,150))
		if not SANDBOX:
			mousex = theta*(210*5)/2 + 204
			mousey = omega*(210)/(-3) + 302
		pygame.draw.circle(screen,RED,(min(max(int(mousex),53),347),min(max(int(mousey),153),447)),3)
		pygame.draw.rect(screen,WHITE,(450,150,300,300))

		font = pygame.font.SysFont("Comic Sans MS",64)
		title = 'Cartpole Tracker'
		text = font.render(title,False,BLACK)
		size = font.size(title)
		screen.blit(text,((DISPLAY_WIDTH-size[0])/2,25))

		nufont = pygame.font.SysFont("Comic Sans MS",24)
		nutitle = 'Phase Space Diagram'
		nutext = nufont.render(nutitle,False,BLACK)
		nusize = nufont.size(nutitle)
		pygame.draw.rect(screen,BLACK,(((300-nusize[0])/2)+50-4,(25 + 300 + 150)-4,nusize[0]+8,nusize[1]+8))
		pygame.draw.rect(screen,WHITE,(((300-nusize[0])/2)+50-2,(25 + 300 + 150)-2,nusize[0]+4,nusize[1]+4))
		screen.blit(nutext,(int((300-nusize[0])/2)+50,(25 + 300 + 150)))

		nufont = pygame.font.SysFont("Comic Sans MS",24)
		nutitle = 'Cartpole Screen'
		nutext = nufont.render(nutitle,False,BLACK)
		nusize = nufont.size(nutitle)
		pygame.draw.rect(screen,BLACK,(((300-nusize[0])/2)+450-4,(25 + 300 + 150)-4,nusize[0]+8,nusize[1]+8))
		pygame.draw.rect(screen,WHITE,(((300-nusize[0])/2)+450-2,(25 + 300 + 150)-2,nusize[0]+4,nusize[1]+4))
		screen.blit(nutext,(int((300-nusize[0])/2)+450,(25 + 300 + 150)))

		pygame.draw.line(screen,BLACK,(450,325),(750,325))
		pygame.draw.rect(screen,BLACK,(575,300,50,25))
		pygame.draw.line(screen,BLACK,(600,300),(ending_pointx,ending_pointy),5)
		if arrow:
			pygame.draw.polygon(screen,RED,[arrow_anchor,arrow1,arrow2,arrow3,arrow4,arrow5,arrow6])




		pygame.display.flip()


		if not SANDBOX:

			pygame.image.save(window,CURRENT_DIR + f'/tracking_pole/markov {i}.png')
			i += 1
			# action = 1 if theta<0 else 0  #SUICIDE
			# action = np.random.randint(0,2) #RANDOM
			# action = 1 if theta>0 else 0#HEURISTIC ONE

			# HEURISTIC TWO
			# fall_time = calc_fall_time(observation)
			# exit_time = calc_exit_time(observation)
			
			# if fall_time<exit_time:
			# 	action = 1 if observation[3]>0 else 0
			# else:
			# 	action = 1 if observation[1]<0 else 0
			
			# action = model.predict(observation[2:].reshape(1,-1))[0] SVM

			s,action = agent.take_action(observation)


			observation, reward, done, info = env.step(action)


			if done:

				crashed = True

		if pygame.event.peek().type==0:
			pygame.event.post(pygame.event.Event(4))

