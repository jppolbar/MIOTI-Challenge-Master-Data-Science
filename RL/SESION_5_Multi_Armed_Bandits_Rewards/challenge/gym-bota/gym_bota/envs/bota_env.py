import numpy as np
import gym
import random

import random
import turtle

from gym.spaces import Discrete


class Bota(gym.Env):

    def __init__(self, human=False, env_info={'state_space':None}):
        super(Bota, self).__init__()

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(13)

        self.llama = "assets/llama_2.gif"
        self.axe = 'assets/axe.gif'

        self.screen_width = 900
        self.screen_length = 400
        self.done = False

        self.scorecount = 0
        self.reward = 0
        self.triggered = False
        self.count = 20
        self.direction = 1
        self.speed = 5
        self.up_down_count = 20
        self.t_rex_speed = 5
        self.obs_size = 4
        self.counter = 100

        self.start = -180
        self.end = -90
        self.diff = (self.end - self.start) / (self.obs_size - 1)

        self.color = ['orange', 'red', 'gray', 'green', 'yellow', 'blue', 'purple', 'magenta']
        # self.color = ['red', 'red', 'red', 'red', 'red', 'red', 'red', 'red']
        self.obs_height = [self.start + self.diff * i for i in range(self.obs_size)]
        self.obs = [turtle.Turtle() for i in range((self.obs_size * 3) // 2)]

        self.resetObs()

        self.done = 0
        self.reward = 0

        # Set up Background
        self.win = turtle.Screen()
        self.win.addshape(self.llama)
        self.win.addshape(self.axe)
        self.win.title('Bota con la Llama')
        self.win.bgcolor('black')
        self.win.setup(width=self.screen_width, height=self.screen_length)
        self.win.tracer(0)

        # T rex config
        self.t_rex = turtle.Turtle()
        self.inializeTrext()

        # Obstacle config
        self.obstacle_speed = -3

        # -------------------- Keyboard control ----------------------

        #         self.win.listen()
        #         self.win.onkey(self.triggerjump, 'space')

        self.score = turtle.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 160)
        self.score.write("Total Dodged : {}".format(self.scorecount), align='center', font=('Courier', 24, 'bold'))

    def inializeTrext(self):

        self.t_rex.shape(self.llama)  # Select a square shape

        self.t_rex.speed(0)
        self.t_rex.shapesize(stretch_wid=1.6, stretch_len=1.6)  # Establecemos longitud
        self.t_rex.penup()
        self.t_rex.color('white')  # Set the color to white
        self.t_rex.setx(-350)
        self.t_rex.sety(-175)

    def resetObs(self):
        for i in self.obs:
            i.flag = False
            i.passed = False
            i.speed(0)
            #             i.shapesize(stretch_wid=1.6, stretch_len=1.6)
            #             i.shape(self.axe)
            i.shape('circle')  # Forma de circulo
            i.color(random.choice(self.color))
            i.penup()
            i.goto(self.screen_width / 2, 0)

    def resetTrex(self):

        self.triggered = False
        self.count = self.up_down_count
        self.direction = 1
        self.t_rex.goto(-350, -175)

    def triggerjump(self):

        self.triggered = True

    def resetScore(self):

        #         self.scorecount = 0
        self.updateScore()

    def updateScore(self):

        self.score.clear()
        self.score.write("Total Dodged : {}".format(self.scorecount), align='center',
                         font=('Courier', 24, 'bold '))

    def resetOb(self, obs):

        obs.flag = False
        obs.passed = False
        obs.setx(self.screen_width / 2)

    def jump(self):

        if self.triggered:
            tex_y = self.t_rex.ycor()
            if self.count > 0:
                self.t_rex.sety(tex_y + self.direction * self.t_rex_speed)
                self.count -= 1
            elif self.count == 0 and self.direction == 1:
                self.count = self.up_down_count
                self.direction = -1
            elif self.count == 0 and self.direction == -1:
                self.triggered = False
                self.count = self.up_down_count
                self.direction = 1

    def move_previous_obstacles(self):

        for i in self.obs:

            if i.flag:

                if i.xcor() + self.obstacle_speed < -1 * self.screen_width / 2:
                    self.resetOb(i)

                else:
                    if abs(self.t_rex.xcor() - i.xcor()) <= 17 and abs(self.t_rex.ycor() - i.ycor()) <= 25:
                        self.done = True
                        self.reset()
                    elif not i.passed and self.t_rex.xcor() > i.xcor():
                        self.scorecount += 1
                        i.passed = True
                        i.setx(i.xcor() + self.obstacle_speed)
                        self.reward += 7
                        self.updateScore()
                    else:
                        i.setx(i.xcor() + self.obstacle_speed)

    def run_frame(self):

        self.win.update()
        self.move_previous_obstacles()
        if self.counter % 60 == 0:
            r1 = random.randint(0, 4)
            if r1 != 0:
                for i in self.obs:
                    if not i.flag:
                        i.flag = True
                        i.sety(random.choice(self.obs_height))
                        break
                self.counter = 1

        else:
            self.counter += 1

        self.jump()

    # ------------------------ AI control ------------------------

    # 0 no hace nada
    # 1 Salta

    def reset(self):
        self.resetObs()
        self.resetTrex()
        self.resetScore()
        state = [i.xcor() * .01 for i in self.obs] + [i.ycor() * .01 for i in self.obs] + [self.t_rex.ycor() * .01]
        return state

    def step(self, action):

        self.reward = 0
        self.done = 0

        if action == 1:  # Salta
            self.reward -= 1
            self.triggerjump()

        self.run_frame()
        self.reward += .1  # Por step dado en el juego

        # Reconpensa por finalización del juego.
        if self.done:
            self.reward -= 20

        state = [i.xcor() * .01 for i in self.obs] + [i.ycor() * .01 for i in self.obs] + [self.t_rex.ycor() * .01]
        return self.reward, state, self.done, self.scorecount

