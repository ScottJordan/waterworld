import numpy as np
from world import World
import pygame
from pygame.constants import K_w, K_a, K_d, K_s, K_SPACE
import time
import sys

game = World(
    map_size=(25, 25),
    width=1024,
    height=512,
    resolution=1,
    reslidar=180,
    rand_walls=False,
    num_items=20,
    move_items=True,
    color=True,
    human_view=True)
game.init()
game.reset()

pygame.init()
screen = pygame.display.set_mode((game.width, game.height), 0, 32)
clock = pygame.time.Clock()
action_map = {
    K_w: 'forward',
    K_a: 'left',
    K_d: 'right',
    K_s: 'backward',
    K_SPACE: 'NOOP'
}


def _handle_player_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if (event.type == pygame.KEYDOWN):
            key = event.key
            if key in action_map.keys():
                return game.agent.key_map[action_map[key]]
    return game.agent.key_map['NOOP']


for i in range(0, 50000):
    dt = 0.  #game.clock.tick_busy_loop(60)
    action = _handle_player_events()
    game.take_action(action)
    sarr = pygame.surfarray.pixels3d(screen)
    #sarr[..., 0] = 255 * game.gscreen
    #sarr[..., 1] = 255 * game.gscreen
    #sarr[..., 2] = 255 * game.gscreen
    sarr[...] = 255. * game.gscreen
    del sarr
    pygame.display.update()

    time.sleep(.1)
