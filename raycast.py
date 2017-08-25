import numpy as np
import pdb


class Raycaster(object):
    def __init__(self,
                 width,
                 height,
                 resX,
                 resL,
                 colormap=None,
                 objectmap=None):
        self.width = width
        self.height = height
        self.resX = resX
        self.resL = resL
        self.resolution = resX
        self.color_map = colormap
        self.object_map = objectmap
        self.eps = 1e-7

    def lidar(self, agent, worldmap):
        #self.map_ = worldmap
        pos = agent.pos.reshape(1, 2)
        dir = agent.dir.reshape(1, 2)
        plane = agent.plane.reshape(1, 2)

        nsamples = self.resL
        cameraX = np.arange(0.0, nsamples, 1.)[:, np.newaxis]
        cameraX = 2.0 * cameraX / float(nsamples) - 1.0
        dists = np.ones((nsamples * 2, ))
        dirs = [dir, dir * -1]
        idx = [range(nsamples), range(nsamples, nsamples * 2)[::-1]]
        for i in range(2):
            rpos, rdir, sdist, ddist, map_, step = self.setup_rays(
                cameraX, pos, dirs[i], plane)
            sdist, ddist, map_, side = self._DDA(
                sdist, ddist, map_, step,
                nonwall=True)

            perpWallDistX = (map_[:, 0] - rpos[:, 0] +
                             (1.0 - step[:, 0]) / 2.0)
            perpWallDistY = (map_[:, 1] - rpos[:, 1] +
                             (1.0 - step[:, 1]) / 2.0)

            scaleX = (perpWallDistX / (rdir[:, 0] + self.eps))[:, np.newaxis]
            scale = (perpWallDistY / (rdir[:, 1] + self.eps))[:, np.newaxis]
            scale[side == 0] = scaleX[side == 0]

            dists[idx[i]] = np.linalg.norm(scale * rdir, axis=1)
        #dists[-10:] = dists[:10]
        return dists

    def setup_rays(self, cameraX, pos, dir, plane):
        ray_pos = np.tile(pos, [cameraX.shape[0], 1])
        ray_dir = dir + plane * cameraX
        map_ = ray_pos.astype(int)

        ray_pow = np.power(ray_dir, 2.0) + self.eps
        ray_div = ray_pow[:, 0] / ray_pow[:, 1]
        delta_dist = np.sqrt(1.0 + np.array([1.0 / ray_div, ray_div])).T

        step = np.ones(ray_dir.shape).astype(int)
        step[ray_dir[:, 0] < 0, 0] = -1
        step[ray_dir[:, 1] < 0, 1] = -1

        side_dist = (map_ + 1.0 - ray_pos) * delta_dist
        _value = (ray_pos - map_) * delta_dist
        side_dist[ray_dir[:, 0] < 0, 0] = _value[ray_dir[:, 0] < 0, 0]
        side_dist[ray_dir[:, 1] < 0, 1] = _value[ray_dir[:, 1] < 0, 1]

        return ray_pos, ray_dir, side_dist, delta_dist, map_, step

    def draw(self, agent, worldmap, nonwall=False):
        #self.map_ = worldmap
        self.pos = agent.pos.reshape(1, 2)
        self.dir = agent.dir.reshape(1, 2)
        self.plane = agent.plane.reshape(1, 2)
        #N = width/resolution
        #N,2
        cameraX = np.arange(0.0, self.width, self.resolution)[:, np.newaxis]
        cameraX = 2.0 * cameraX / float(self.width) - 1.0

        #set the rayPos to the players current position
        ray_pos = np.tile(self.pos, [cameraX.shape[0], 1])  #N,2

        #ray direction
        ray_dir = self.dir + self.plane * cameraX  # N,2
        #which box of the map we're in
        map_ = ray_pos.astype(int)

        ray_pow = np.power(ray_dir, 2.0) + self.eps
        ray_div = ray_pow[:, 0] / (ray_pow[:, 1])
        delta_dist = np.sqrt(1.0 + np.array([1.0 / (ray_div), ray_div
                                             ])).T  #N,2

        # N,2
        step = np.ones(ray_dir.shape).astype(int)
        step[ray_dir[:, 0] < 0, 0] = -1
        step[ray_dir[:, 1] < 0, 1] = -1

        # N,2
        side_dist = (map_ + 1.0 - ray_pos) * delta_dist
        _value = (ray_pos - map_) * delta_dist

        side_dist[ray_dir[:, 0] < 0, 0] = _value[ray_dir[:, 0] < 0, 0]
        side_dist[ray_dir[:, 1] < 0, 1] = _value[ray_dir[:, 1] < 0, 1]

        side_dist, delta_dist, map_, side = self._DDA(side_dist, delta_dist,
                                                      map_, step)

        perpWallDistX = (map_[:, 0] - ray_pos[:, 0] + (1.0 - step[:, 0]) / 2.0)
        perpWallDistX = perpWallDistX / (ray_dir[:, 0] + self.eps)
        perpWallDistX = perpWallDistX[:, np.newaxis]

        perpWallDistY = (map_[:, 1] - ray_pos[:, 1] + (1.0 - step[:, 1]) / 2.0)
        perpWallDistY = perpWallDistY / (ray_dir[:, 1] + self.eps)
        perpWallDistY = perpWallDistY[:, np.newaxis]

        perpWallDist = perpWallDistY
        perpWallDist[side == 0] = perpWallDistX[side == 0]

        scale = 1.0
        lineHeights = ((self.height / (perpWallDist + self.eps)) * scale)
        lineHeights = lineHeights.astype(int)

        tops = -(lineHeights) / 2.0 + self.height / 2.0
        tops[tops < 0] = 0.0
        tops = tops.astype(int)

        bottoms = lineHeights / 2.0 + self.height / 2.0
        bottoms[bottoms >= self.height] = self.height - 1
        bottoms = bottoms.astype(int)

        #visible_blocks = self.map_[map_[:, 0], map_[:, 1]]
        #coloring = np.ones((bottoms.shape[0], 3)) * 255.0
        c = self.color_map[self.object_map['wall']]
        coloring = np.tile(c, [bottoms.shape[0], 1])
        #for k in self.object_map.keys():
        #    if self.color_map[self.object_map[k]] != None:
        #        c = self.color_map[self.object_map[k]]
        #        sel = visible_blocks == self.object_map[k]
        #        coloring[sel] = np.tile(c, [bottoms.shape[0], 1])[sel]

        shading = np.abs(perpWallDist * 1) * 1.5
        coloring = coloring - shading
        coloring = np.clip(coloring, 0, 255)
        coloring[(side == 1.0).flatten(), :] *= 0.65  #lighting apparently

        cameraX = np.arange(0, self.width, self.resolution)
        returns = [cameraX, tops, bottoms, coloring]

        return [r.astype(int) for r in returns] + [perpWallDist]

    def _DDA(self, side_dist, delta_dist, map_, step, nonwall=False):
        #tested against for-loop version using line_profiler
        #for-loop take about 0.005968s per call
        #this version takes 0.000416s per call
        hits = np.zeros((map_.shape[0], 1))
        side = np.zeros((map_.shape[0], 1))

        while np.sum(hits) < side_dist.shape[0]:
            #only update values that havent hit a wall. So are 0 still.

            update_mask = np.logical_not(hits).astype(np.bool)

            # 1 => 1, 0
            # 0 => 0, 1
            mask = (side_dist[:, 0] < side_dist[:, 1])[:, np.newaxis]

            sel = (update_mask & (mask == True)).flatten()
            side_dist[sel, 0] += delta_dist[sel, 0]
            map_[sel, 0] += step[sel, 0]
            side[sel] = np.zeros(side.shape)[sel]

            sel = (update_mask & (mask == False)).flatten()
            side_dist[sel, 1] += delta_dist[sel, 1]
            map_[sel, 1] += step[sel, 1]
            side[sel] = np.ones(side.shape)[sel]

            #once it becomes 1 it never goes back to 0.
            if nonwall:
                hits = np.logical_or(hits, (
                    self.map_[map_[:, 0], map_[:, 1]] > 0)[:, np.newaxis])
            else:
                hits = np.logical_or(hits, (
                    self.map_[map_[:, 0], map_[:, 1]] == 1)[:, np.newaxis])

        return side_dist, delta_dist, map_, side

    def draw_sprites(self, agent, ipos):
        pos = agent.pos
        plane = agent.plane
        dir = agent.dir

        spriteX = ipos[:, 0] - pos[0]
        spriteY = ipos[:, 1] - pos[1]

        invDet = 1.0 / (plane[0] * dir[1] - dir[0] * plane[1])
        transX = invDet * (dir[1] * spriteX - dir[0] * spriteY)
        transY = invDet * (-plane[1] * spriteX + plane[0] * spriteY)
        eps = self.eps
        spritescreenX = ((self.width / 2) * (1 + transX /
                                             (transY + eps))).astype(int)

        spriteHeight = np.abs((self.height / (transY + eps)).astype(int)) / 2

        drawStartY = -spriteHeight / 2 + self.height / 2
        drawStartY[drawStartY < 0] = 0
        drawEndY = spriteHeight / 2 + self.height / 2
        drawEndY[drawEndY >= self.height] = self.height - 1

        spriteWidth = np.abs((self.width / (transY + eps)).astype(int)) / 2

        drawStartX = -spriteWidth / 2 + spritescreenX
        #drawStartX[drawStartX < 0] = 0
        drawEndX = spriteWidth / 2 + spritescreenX
        #drawEndX[drawEndX >= self.width] = self.width - 1

        drawX = np.array([drawStartX, drawEndX]).T
        drawY = np.array([drawStartY, drawEndY]).T
        trans = np.array([transX, transY]).T
        sdim = np.array([spriteWidth, spriteHeight]).T

        return drawX, drawY, trans, sdim, spritescreenX
