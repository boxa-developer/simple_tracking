import numpy as np
from collections import OrderedDict
from loguru import logger


class Tracker:
    def __init__(self):
        self.vehicles = OrderedDict()
        self.lives = OrderedDict()
        self.tails = OrderedDict()
        self.nextId = 0

    def register(self, cord):
        logger.info(f'Car registered with ID:{self.nextId}')
        self.vehicles[self.nextId] = cord
        self.lives[self.nextId] = 20
        self.tails[self.nextId] = [self.calc_centroid(cord)]
        self.nextId += 1

    def deregister(self, track_id):
        del self.vehicles[track_id]
        del self.lives[track_id]

    @staticmethod
    def calc_distance(c1, c2):
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    @staticmethod
    def calc_centroid(cords):
        x1, y1, x2, y2 = cords
        return (x2 + x1) // 2, (y1 + y2) // 2

    def is_in(self, c1, c2):
        cnt1 = self.calc_centroid(c1)
        cnt2 = self.calc_centroid(c2)
        height = c1[3]-c1[1]
        if self.calc_distance(cnt1, cnt2) < height*0.2:
            return True
        return False

    def update(self, cords):
        if len(self.vehicles) > 0:
            updated = []
            used_cords = []
            for ids, vehicle_cord in self.vehicles.items():
                for i, new_cord in enumerate(cords):
                    if self.is_in(new_cord, vehicle_cord):
                        self.vehicles[ids] = new_cord
                        self.tails[ids].append(self.calc_centroid(new_cord))
                        used_cords.append(i)
                        updated.append(ids)

            for i, new_cord in enumerate(cords):
                if i not in used_cords:
                    self.register(new_cord)

            for ids in set(self.vehicles.keys()) - set(updated):
                if self.lives[ids] > 0:
                    self.lives[ids] -= 1
                else:
                    self.deregister(ids)
        else:
            logger.info(f'Vehicles Not Found Starting registering {len(cords)} cars')
            for cord in cords:
                self.register(cord)
        return self.vehicles, self.tails
