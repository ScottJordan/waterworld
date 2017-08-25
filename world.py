#import pygame
from PIL import Image
import numpy as np
from raycast import Raycaster
from skimage.draw import line_aa
#from pygame.constants import K_w, K_a, K_d, K_s, K_SPACE
import sys
import pdb

OBJECT_MAP = {
    'open': 0,
    'wall': 1,
    'food': 2,
    'poison': 3,
    'agent': 4,
    'floor': 5,
    'sky': 6
}

COLOR_MAP = {
    OBJECT_MAP['wall']: (93., 192., 221.),  # pretty blue color
    OBJECT_MAP['food']: (255., 0., 0.),
    OBJECT_MAP['poison']: (0., 255., 0.),
    OBJECT_MAP['floor']: (255., 255., 255.),
    OBJECT_MAP['sky']: (0., 0., 0.),
    OBJECT_MAP['agent']: (0., 0., 255.),
    OBJECT_MAP['open']: None
}

REWARD_MAP = {
    OBJECT_MAP['wall']: -0.1,
    OBJECT_MAP['food']: 1.0,
    OBJECT_MAP['poison']: -1.0
}

ITEM_AGE_LIMIT = 1000
np.set_printoptions(4, suppress=True)
COLOR_KEY = (0, 0, 0)
TEX_WIDTH = 32
TEX_HEIGHT = 32
CONVERT = (np.array([0.2989, 0.5870, 0.1140])).astype(np.float32)


class Agent(object):
    def __init__(self,
                 pos,
                 dir,
                 plane,
                 max_move=.25,
                 max_rot=(np.pi / 16),
                 max_acc=0.3):
        self.pos = np.copy(pos)
        self.dir = np.copy(dir)
        self.plane = np.copy(plane)
        self.type = OBJECT_MAP['agent']
        self.max_move = max_move
        self.max_rot = max_rot
        self.max_acc = max_acc
        self.vel = np.array([0., 0.])
        self.action_set = {
            0: (max_move, 0.),
            1: (0., -max_rot),
            2: (0., max_rot),
            3: (-max_move, 0.),
            4: (0., 0.)
        }
        self.key_map = {
            'forward': 0,
            'left': 1,
            'right': 2,
            'backward': 3,
            'NOOP': 4
        }
        #self.event_map = {K_w: 0, K_a: 1, K_d: 2, K_s: 3, K_SPACE: 4}


class World(object):
    def __init__(self,
                 map_size,
                 width=48,
                 height=48,
                 resolution=1,
                 reslidar=180,
                 rand_walls=False,
                 num_items=10,
                 move_items=False,
                 color=False,
                 human_view=True):
        self.map_size = map_size
        self.width = width
        self.height = height
        self.resolution = resolution
        self.reslidar = reslidar
        self.run_lidar = False
        if reslidar > 0:
            self.run_lidar = True
        self.random_walls = rand_walls
        self.num_items = num_items
        self.move_items = move_items
        self.color = color
        self.human_view = human_view
        self.init_pos = np.array([1., 1.])
        self.init_dir = np.array([1., 0.])
        self.init_angle = 0.
        self.init_plane = np.array([0.0, 0.66])
        self.agent = Agent(self.init_pos, self.init_dir, self.init_plane)
        self.imnum = 0
        rw = self.width
        if self.human_view:
            rw = self.width / 2

        self.raycaster = Raycaster(rw, self.height, self.resolution, reslidar,
                                   COLOR_MAP, OBJECT_MAP)
        self.lidar = np.zeros(reslidar * 2)
        self.do_step = True
        self.sprites = None
        self.minisprites = None
        cscale = .75
        bgcolor = np.array(
            [93. * cscale, 192. * cscale, 221. * cscale],
            dtype=np.float32) / .255

        self.load_images()

        if (self.color):
            self.gscreen = np.zeros(
                (self.width, self.height, 3), dtype=np.float32)
            self.background = np.zeros_like(self.gscreen)
            #self.background[:, :, 0] = bgcolor[0] # just solid color
            #self.background[:, :, 1] = bgcolor[1]
            #self.background[:, :, 2] = bgcolor[2]
            if self.human_view:
                self.background[:int(self.width / 2), :, :] = np.copy(self.bg1[
                    0])
                self.background[int(self.width / 2):, :, :] = np.copy(self.bg2[
                    0])
            else:
                self.background[:] = np.copy(self.bg1[0])
        else:
            self.gscreen = np.zeros(
                (self.width, self.height), dtype=np.float32)
            self.background = np.zeros_like(self.gscreen)
            #self.background[:] = np.dot(bgcolor, CONVERT) #solid color
            if self.human_view:
                self.background[:int(self.width / 2), :] = np.copy(self.bg1[2])
                self.background[int(self.width / 2):, :] = np.copy(self.bg2[2])
            else:
                self.background[:] = np.copy(self.bg1[2])
        self.wbackground = np.copy(self.background)

    def load_images(self):
        files = ["./sprites/dolphin.png", "./sprites/subfix.png"]
        if self.sprites is None:
            self.sprites = {
                OBJECT_MAP['poison']: loadimage(
                    files[0],
                    TEX_WIDTH,
                    TEX_HEIGHT,
                    color=self.color,
                    color_key=(0, 0, 0)),
                OBJECT_MAP['food']: loadimage(
                    files[1],
                    TEX_WIDTH,
                    TEX_HEIGHT,
                    color=self.color,
                    color_key=(95, 183, 255))
            }
            mdim = self.width / 2 / self.map_size[0]
            self.minisprites = {
                OBJECT_MAP['poison']: loadimage(
                    files[0],
                    mdim,
                    mdim,
                    color=self.color,
                    color_key=(0, 0, 0)),
                OBJECT_MAP['food']: loadimage(
                    files[1],
                    mdim,
                    mdim,
                    color=self.color,
                    color_key=(95, 183, 255)),
                OBJECT_MAP['agent']: agent_sprite((0, 0, 0), self.color,
                                                  mdim / 2)
            }
        self.bg1 = loadbackground(self, './backgrounds/ocean-floor.png')
        self.bg2 = loadbackground(self, './backgrounds/ocean-floor-2.jpg')

    def make_bgwall(self):
        self.wbackground = np.copy(self.background)
        xpos = self.width / 2
        ypos = 0
        block_dim = xpos / self.map_size[0]
        block = np.zeros((block_dim, block_dim, 3), np.float32)
        block[:, :, 0] = 93. / 255. * .75,
        block[:, :, 1] = 192. / 255. * .75
        block[:, :, 2] = 211. / 255. * .75
        if not self.color:
            block = block.dot(CONVERT)
        for row in range(self.world_map.shape[0]):
            xs = xpos + (row * block_dim)
            xe = xs + block_dim
            for col in range(self.world_map.shape[1]):
                ys = col * block_dim
                ye = ys + block_dim
                if self.world_map[row, col] == OBJECT_MAP['wall']:
                    if self.color:
                        self.wbackground[xs:xe, ys:ye, :] = np.copy(block)
                    else:
                        self.wbackground[xs:xe, ys:ye] = np.copy(block)

    def gen_basic_walls(self):
        self.world_map[5, 5:11] = OBJECT_MAP['wall']
        self.world_map[20, 5:11] = OBJECT_MAP['wall']
        self.world_map[5:21, 11] = OBJECT_MAP['wall']
        self.world_map[5, 14:20] = OBJECT_MAP['wall']
        self.world_map[20, 14:20] = OBJECT_MAP['wall']
        self.world_map[5:21, 20] = OBJECT_MAP['wall']

    def gen_random_walls(self):
        # generates 4-9 random L-shaped walls
        numPillars = np.random.randint(4, 10)
        # pillarPos = np.random.randint(25, size=(2,numPillars))
        map_x = self.map_size[0]
        map_y = self.map_size[1]
        pillarPosx = np.random.randint(map_x, size=numPillars)
        pillarPosy = np.random.randint(map_y, size=numPillars)
        pillarPos = [pillarPosx, pillarPosy]

        for i in range(numPillars):
            x = pillarPos[0][i]
            y = pillarPos[1][i]
            xlen = np.random.randint(10)
            xdir = np.random.randint(2)
            if xdir == 0:
                self.world_map[x:x + xlen, y] = OBJECT_MAP['wall']
            else:
                self.world_map[x - xlen:x, y] = OBJECT_MAP['wall']
            ylen = np.random.randint(10)
            ydir = np.random.randint(2)
            if ydir == 0:
                self.world_map[x, y:y + ylen] = OBJECT_MAP['wall']
            else:
                self.world_map[x, y - ylen:y] = OBJECT_MAP['wall']

    def take_action(self, action):
        act = self.agent.action_set[action]
        self.step(act)
        return self.reward

    def act(self, action):
        self.items[:, 3] += 1
        agent = self.agent
        oldpos = agent.pos
        olddir = agent.dir
        oldplane = agent.plane

        newpos, newdir, newplane, newvel = self.move(agent, action)
        reward = 0.
        if self.validmove(oldpos, newpos):

            map_pos = newpos.astype(int)
            atXY = self.world_map[map_pos[0], map_pos[1]]
            grab = self.items[:, 4] < 0.25
            self.items[grab, 3] = ITEM_AGE_LIMIT
            for r in self.items[grab, 2]:
                reward += REWARD_MAP[r]

            self.agent.pos = np.copy(newpos)
            self.agent.dir = np.copy(newdir)
            self.agent.plane = np.copy(newplane)
            self.agent.vel = np.copy(newvel)
            oldmap_pos = oldpos.astype(int)
            self.world_map[oldmap_pos[0], oldmap_pos[1]] = OBJECT_MAP['open']
        else:
            reward = -1.
        self.reward = reward

    def cleanup_items(self, agent_pos):
        old = self.items[:, 3] >= ITEM_AGE_LIMIT
        old = np.logical_and(old, np.random.rand(old.shape[0]) < 0.1)
        loc = self.items[old, :2].astype(int)
        self.world_map[(loc[:, 0], loc[:, 1])] = OBJECT_MAP['open']
        num = np.sum(old)
        if num > 0:
            open_spots = np.where(self.world_map == OBJECT_MAP['open'])
            sel = np.random.choice(
                range(open_spots[0].shape[0]), size=num, replace=False)
            self.items[old, 3] = 0
            x, y = open_spots[0][sel], open_spots[1][sel]
            self.items[old, 0] = x + 0.5
            self.items[old, 1] = y + 0.5
            self.world_map[x, y] = self.items[old, 2].astype(int)

    def getGameState(self):
        return self.lidar

    def getScore(self):
        return self.score

    def game_over(self):
        return False

    def init(self):
        # define map and put up border wall
        world_map = np.zeros(self.map_size, dtype=np.int32)
        world_map[:, 0] = OBJECT_MAP['wall']
        world_map[:, -1] = OBJECT_MAP['wall']
        world_map[0, :] = OBJECT_MAP['wall']
        world_map[-1, :] = OBJECT_MAP['wall']
        self.world_map = world_map
        if self.random_walls:
            self.gen_random_walls()
        else:
            self.gen_basic_walls()
        if self.human_view:
            self.make_bgwall()

        open_spots = np.where(self.world_map == OBJECT_MAP['open'])
        sel = np.random.choice(
            range(open_spots[0].shape[0]), size=self.num_items, replace=False)
        poison = np.random.rand(self.num_items) < 0.5
        itype = np.ones(self.num_items, dtype=np.int32) * OBJECT_MAP['food']
        itype[poison] = OBJECT_MAP['poison']

        # directions each item will move
        #   0: +x 
        #   1: -x
        #   2: +y
        #   3: -y
        idirs = np.random.choice(4, self.num_items)

        #items has shape num_itemsx6 with cols (x, y, type, age, distance, direction)
        items = np.zeros((self.num_items, 6))
        items[:, 0] = open_spots[0][sel] + 0.5
        items[:, 1] = open_spots[1][sel] + 0.5
        items[:, 2] = itype
        items[:, 5] = idirs

        self.world_map[items[:, 0].astype(int), items[:, 1].astype(
            int)] = items[:, 2].astype(int)
        self.items = items

        self.score = 0.
        self.reward = 0.
        self.raycaster.map_ = self.world_map

    def reset(self):
        self.init()

        open_spots = np.where(self.world_map == OBJECT_MAP['open'])
        i = np.random.randint(0, open_spots[0].shape[0])
        init_pos = np.array([open_spots[0][i], open_spots[1][i]])

        self.agent.pos = init_pos
        self.agent.dir = self.init_dir
        self.agent.plane = self.init_plane

    def step(self, action):
        self.gscreen[:] = np.copy(self.wbackground)
        self.act(action)
        self.cleanup_items(self.agent.pos)
        self.distitems()
        self.score += self.reward
        # get variables for drawing walls (first person view)
        c, t, b, col, dist = self.raycaster.draw(self.agent, self.world_map)
        if self.run_lidar:
            self.lidar = self.raycaster.lidar(self.agent, self.world_map)
            self.lidar = 1. - (np.clip(self.lidar, 0., 30.) / 30.)
        if not self.color:
            gray = np.dot(col[..., :], CONVERT) / 255.
        else:
            col = col.astype(np.float32) / float(255.)
        for i in range(len(c)):
            vert = np.arange(t[i], b[i], 1).astype(int)
            if self.color:
                self.gscreen[c[i], vert, :] = col[c[i]]
            else:
                self.gscreen[c[i], vert] = gray[c[i]]

        if self.human_view:
            self.draw_topdown()
            if self.run_lidar:
                self.draw_lidar()
        # try moving items
        if self.move_items:
            self.moveitems()

        not_old = self.items[self.items[:, 3] < ITEM_AGE_LIMIT, :]
        drawX, drawY, trans, sdim, ssx = \
                self.raycaster.draw_sprites(self.agent, not_old[:, :2])
        xwidth = self.width
        if self.human_view:
            xwidth = self.width / 2
        for i in range(not_old.shape[0]):
            if (trans[i, 1] > 0.01):
                stype = int(not_old[i, 2])

                x = np.arange(drawX[i, 0], drawX[i, 1], 1).astype(int)
                x = x[x > 0]
                x = x[x < xwidth]
                x = x[(trans[i, 1] < dist[x]).reshape(-1)]
                if x.shape[0] > 0:
                    txX = 256 * (x - (-sdim[i, 0] / 2 + ssx[i]))
                    texX = (txX * TEX_WIDTH / sdim[i, 0]) / 256
                    y = np.arange(drawY[i, 0], drawY[i, 1], 1).astype(int)
                    d = y * 256 - self.height * 128 + sdim[i, 1] * 128
                    texY = ((d * TEX_HEIGHT) / sdim[i, 1]) / 256
                    gsprite = self.sprites[stype][1]
                    drw = self.sprites[stype][0]
                    if self.color:
                        for j in range(x.shape[0]):
                            sel = drw[texX[j], texY, 0]
                            dy = y[sel]
                            ty = texY[sel]
                            self.gscreen[x[j], dy, 0] = gsprite[texX[j], ty, 0]
                            self.gscreen[x[j], dy, 1] = gsprite[texX[j], ty, 1]
                            self.gscreen[x[j], dy, 2] = gsprite[texX[j], ty, 2]
                    else:
                        for j in range(x.shape[0]):
                            sel = drw[texX[j], texY]
                            dy = y[sel]
                            ty = texY[sel]
                            self.gscreen[x[j], dy] = gsprite[texX[j], ty]

    def validmove(self, pos, newpos):
        if (newpos[0] > 0 and \
            newpos[1] > 0) and \
           (newpos[0] < (self.world_map.shape[0]-1) and \
            newpos[1] < (self.world_map.shape[1]-1)):
            at = self.world_map[int(newpos[0]), int(newpos[1])]
            if at == OBJECT_MAP['wall']:
                #print("invallid move at wall")
                return False
            else:
                #print("valid move")
                return True
        else:
            #print("invalid move, out of zone")
            return False

    def move(self, agent, action):
        vel = np.copy(agent.vel)
        pos = np.copy(agent.pos)
        dir = np.copy(agent.dir)
        plane = np.copy(agent.plane)

        #acc = action[0]
        #vel *= 0.9
        try:
            xturn = np.cos(action[1])
            yturn = np.sin(action[1])
        except TypeError:
            pdb.set_trace()
        dirx = dir[0] * xturn - dir[1] * yturn
        diry = dir[0] * yturn + dir[1] * xturn

        planex = plane[0] * xturn - plane[1] * yturn
        planey = plane[0] * yturn + plane[1] * xturn

        dir[0] = dirx
        dir[1] = diry
        plane[0] = planex
        plane[1] = planey

        vel = dir * action[0]
        nvel = np.copy(vel)
        step = 0.95
        count = 0
        while not self.validmove(agent.pos, pos + nvel):
            if count > 500:
                self.reset()
                #print("got stuck in loop reseting")
                return self.agent.pos, self.agent.dir, self.agent.plane, np.array(
                    [0., 0])
            nvel = vel * step
            step -= 0.05
            count += 1

        npos = pos + nvel
        mapp = npos.astype(int)
        if self.world_map[mapp[0] + 1, mapp[1]] == OBJECT_MAP['wall']:
            nvel[0] = 0.
        elif self.world_map[mapp[0], mapp[1] + 1] == OBJECT_MAP['wall']:
            nvel[1] = 0.
        elif self.world_map[mapp[0] + 1, mapp[1] + 1] == OBJECT_MAP['wall']:
            nvel = np.array([0., 0.])
        else:
            pass
        return npos, dir, plane, nvel

    def distitems(self):
        pos = self.agent.pos
        ipx = self.items[:, 0]
        ipy = self.items[:, 1]
        old = self.items[:, 3] >= ITEM_AGE_LIMIT
        self.items[:, 4] = (ipx - pos[0])**2 + (ipy - pos[1])**2
        self.items[old, 4] = 1e5
        self.items = self.items[self.items[:, 4].argsort()[::-1], :]

    def moveitems(self):
        if self.imnum % 3 != 0:
            self.imnum += 1
            return
        self.imnum += 1
        direction_map_x = {0: 1, 1: -1, 2: 0, 3: 0}

        direction_map_y = {0: 0, 1: 0, 2: 1, 3: -1}

        for item_it in self.items:
            itdir_tmp = item_it[5]
            ittyp_tmp = item_it[2]
            oldpos = [item_it[0], item_it[1]]
            newposX = item_it[0] + direction_map_x[itdir_tmp]
            newposY = item_it[1] + direction_map_y[itdir_tmp]
            newpos = [newposX, newposY]
            if self.valid_itemmove(oldpos, newpos) == False:
                item_it[5] = np.random.randint(4)
            else:
                self.world_map[oldpos[0], oldpos[1]] = OBJECT_MAP['open']
                self.world_map[newpos[0], newpos[1]] = ittyp_tmp
                item_it[0] = newpos[0]
                item_it[1] = newpos[1]

    def draw_topdown(self):
        # start drawing the top-down view
        xpos = self.width / 2
        ypos = 0
        block_dim = xpos / self.map_size[0]
        for row in self.world_map:
            for col in row:
                if col == OBJECT_MAP['food'] or col == OBJECT_MAP['poison']:
                    self.draw_mini(xpos, ypos, col)
                ypos = ypos + block_dim
            xpos = xpos + block_dim
            ypos = 0

        xpos = self.agent.pos[0] * block_dim + self.width / 2.
        ypos = self.agent.pos[1] * block_dim
        self.draw_mini(xpos, ypos, OBJECT_MAP['agent'])

    def draw_lidar(self):
        xpos = self.width / 2
        block_dim = xpos / self.map_size[0]

        pos = self.agent.pos.reshape(1, 2)
        dir = self.agent.dir.reshape(1, 2)  #* 0.1
        plane = self.agent.plane.reshape(1, 2)  #* 4

        nsamples = self.reslidar
        lidar = ((1. - self.lidar) * 30.).astype(int) * block_dim
        cameraX = np.arange(0.0, nsamples, 1.)[:, np.newaxis]
        cameraX = 2.0 * cameraX / float(nsamples) - 1.0
        rdirs = np.ones((nsamples * 2, 2))
        dirs = [dir, dir * -1]
        for i in range(2):
            rdirs[(nsamples * i):(nsamples * (i + 1)), :] = dirs[
                i] + plane * cameraX

        xpos = self.agent.pos[0] * block_dim  #+ self.width / 2.
        ypos = self.agent.pos[1] * block_dim
        wd, ht = sprite = self.minisprites[OBJECT_MAP['agent']][1].shape[:2]
        xpos += wd / 2 - 1
        ypos += ht / 2 - 1

        pt = ((rdirs * lidar[:, np.newaxis]) + [xpos, ypos]).astype(np.int)
        #limg = np.zeros((self.width / 2., self.width / 2.))
        #r = []
        #c = []
        #v = []
        for i in range(pt.shape[0]):
            rr, cc, val = line_aa(int(xpos), int(ypos), pt[i, 0], pt[i, 1])
            rr = np.clip(rr, 0, 511)
            cc = np.clip(cc, 0, 511)
            #val *= 255
            #limg[rr, cc] += val
            #r.append(rr)
            #c.append(cc)
            #v.append(val)
            #print(rr.shape, cc.shape, val.shape)
            row = rr + int(self.width / 2)
            self.gscreen[row, cc, 0] = val
            self.gscreen[row, cc, 1] = val
            self.gscreen[row, cc, 2] = val

        #return r, c, v

    def draw_mini(self, x, y, obj):
        sprite = self.minisprites[obj]
        drw = sprite[0]
        img = sprite[1]
        (wd, ht) = drw.shape[:2]
        if self.color:
            self.gscreen[x:x + wd, y:y + ht, :][drw] = img[drw]
        else:
            self.gscreen[x:x + wd, y:y + ht][drw] = img[drw]

    def valid_itemmove(self, oldpos, newpos):
        if (newpos[0] > 0 and \
            newpos[1] > 0) and \
           (newpos[0] < (self.world_map.shape[0]-1) and \
            newpos[1] < (self.world_map.shape[1]-1)):
            at = self.world_map[int(newpos[0]), int(newpos[1])]
            if at == OBJECT_MAP['wall']:
                #print("invallid move at wall")
                return False
            elif at == OBJECT_MAP['food'] or \
                    at == OBJECT_MAP['poison']:
                return False
                # invalid to move into another actor
            else:
                #print("valid move")
                return True
        else:
            #print("invalid move, out of zone")
            return False


def getDrawableC(surf, ck):
    sarr = np.array(surf)
    drw = np.ones((sarr.shape[0], sarr.shape[1], 3), dtype=bool)
    if ck is None:
        return drw
    for i in range(sarr.shape[0]):
        for j in range(sarr.shape[1]):
            if (sarr[i, j, 0] == ck[0] and sarr[i, j, 1] == ck[1] and
                    sarr[i, j, 2] == ck[2]):
                drw[i, j, :] = False
    return drw


def getDrawable(surf, ck):
    sarr = np.array(surf)
    drw = np.ones((sarr.shape[0], sarr.shape[1]), dtype=bool)
    if ck is None:
        return drw
    for i in range(sarr.shape[0]):
        for j in range(sarr.shape[1]):
            if (sarr[i, j, 0] == ck[0] and sarr[i, j, 1] == ck[1] and
                    sarr[i, j, 2] == ck[2]):
                drw[i, j] = False
    return drw


def getGraySurf(surf):
    sarr = np.array(surf, np.float32)
    gray = np.dot(sarr[..., :3], CONVERT).astype(np.float32) / 255.
    return gray


def agent_sprite(col, color, radius):
    drw = np.zeros((radius * 2, radius * 2), dtype=np.bool)
    for i in range(radius * 2):
        for j in range(radius * 2):
            d = np.sqrt((i - radius)**2 + (j - radius)**2)
            if d <= radius:
                drw[i, j] = True
    img = np.zeros((radius * 2, radius * 2), dtype=np.float32)
    img[drw] = np.dot(col, CONVERT)
    if color:
        img = np.zeros((radius * 2, radius * 2, 3), dtype=np.float32)
        dr2 = np.zeros((radius * 2, radius * 2, 3), dtype=np.bool)
        for i in range(3):
            dr2[:, :, i] = np.copy(drw)
            img[:, :, i][drw] = col[i]

        drw = np.copy(dr2)
    return [drw, img]


def loadimage(file, width, height, color=True, color_key=None):
    image = Image.open(file)
    scaled = image.resize((width, height))
    if color:
        drw = getDrawableC(scaled, color_key)
        drw = np.transpose(drw, axes=[1, 0, 2])
        img = np.array(scaled, np.float32) / 255.
        img = np.transpose(img, axes=[1, 0, 2])[:, :, :3]
    else:
        drw = getDrawable(scaled, color_key).T
        img = getGraySurf(scaled).T
    return [drw, img]


def loadbackground(world, file, color_key=None):
    bg = Image.open(file)
    if world.human_view:
        bdim = int(world.width / 2)
        bg = bg.resize((bdim, world.height))
    else:
        bg = bg.resize((world.width, world.height))
    drw = getDrawable(bg, color_key).T
    gray = getGraySurf(bg).T
    cbg = np.array(bg, dtype=np.float32) / 255.
    cbg = np.transpose(cbg, axes=[1, 0, 2])[:, :, :3]
    return [cbg, drw, gray]
