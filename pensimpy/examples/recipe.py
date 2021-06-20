import json
import math

from typing import Dict, List, Tuple


class Setpoint:
    """
    Encapsulate a value at a point of time
    """
    def __init__(self, time: float, value: float):
        self.time = time
        self.value = value

    def get_sp_dict(self):
        return {"time": self.time, "value": self.value}

    def __str__(self):
        return json.dumps(self.get_sp_dict())


class Recipe:
    """
    A store for a series of setpoints [{"time": , "value": v}...]
    """
    def __init__(self, sp_list: List[Dict[str, float]],
                 name: str):
        assert len(sp_list) > 0, "Can't initiate a recipe instance with an empty setpoint list"
        assert isinstance(name, str), "Can't initiate a recipe instance without a name"
        self.name = name
        self._sp_list = []
        self.index_lookup = {}
        self.add_setpoints(sp_list)
        dict()

    @property
    def sp_list(self):
        return self._sp_list

    @sp_list.setter
    def sp_list(self, val: List[Setpoint]):
        self._sp_list = sorted(val, key=lambda x: x.time)
        self.index_lookup = self.create_index_lookup(self._sp_list)

    @classmethod
    def create_index_lookup(cls, sp_list: List[Setpoint]) -> Dict:
        return dict([(sp.time, i) for i, sp in enumerate(sp_list)])

    def add_setpoint(self, sp: Dict[str, float]):
        sp = Setpoint(**sp)
        assert sp.time not in self.index_lookup.keys(), f"time = {sp.time} already exists"
        self.sp_list = self.sp_list + [sp]

    def add_setpoints(self, sp_list: List[Dict[str, float]]):
        for sp in sp_list:
            self.add_setpoint(sp)

    def find_setpoints_interval(self, time: float) -> Tuple[Setpoint, Setpoint]:
        assert len(self.sp_list) > 0, "no setpoints available"

        if len(self.sp_list) == 1:
            return self.sp_list[0], self.sp_list[0]

        left_bound_time = self.sp_list[0].time
        right_bound_time = self.sp_list[-1].time

        # back-fill when pass left bound
        if time < left_bound_time:
            return self.sp_list[0], self.sp_list[0]

        # forward-fill when pass right bound
        if time > right_bound_time:
            return self.sp_list[-1], self.sp_list[-1]

        start = 0
        end = len(self.sp_list) - 1
        while end > start:
            mid = math.ceil((end + start) / 2)
            if time >= self.sp_list[mid].time:
                start = mid
            else:
                end = mid - 1
        if self.sp_list[start].time == time:
            return self.sp_list[start], self.sp_list[start]
        else:
            return self.sp_list[start], self.sp_list[min(start + 1, len(self.sp_list) - 1)]

    def get_value_at(self, time: float) -> float:
        left_sp, right_sp = self.find_setpoints_interval(time)

        # forwarding filling
        return left_sp.value


class RecipeCombo:
    """
    A collection of recipes
    """
    def __init__(self, recipe_dict: Dict[str, Recipe]):
        assert len(recipe_dict) > 0, "Can't initiate a recipe combo instance with an empty recipe dict"
        self.recipe_dict = recipe_dict

    def get_values_dict_at(self, time: float) -> Dict:
        """
        Get value of each recipe at given time
        """
        values_dict = {}
        for name, recipe in self.recipe_dict.items():
            values_dict[name] = recipe.get_value_at(time=time)

        return values_dict