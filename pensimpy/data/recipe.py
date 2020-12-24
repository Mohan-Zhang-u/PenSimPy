import math
from typing import Dict, List, Tuple
from enum import Enum
import json

from pensimpy.data.constants import FLOAT_PRECISION, DEFAULT_MIN_TIME_IN_HOUR, DEFAULT_MAX_TIME_IN_HOUR, \
    MINUTES_IN_AN_HOUR


class ValueType(Enum):
    NORMAL = 1
    LOWER_BOUND = 2
    UPPER_BOUND = 3


class FillingMethod(Enum):
    BACKWARD = 1
    FROWARD = 2
    INTERPOLATION = 3


class Setpoint:
    def __init__(self, time: float, value: float,
                 lower_bound: float = None, upper_bound: float = None):
        self.time = time
        self.value = round(value, FLOAT_PRECISION)
        self._lower_bound = self.value if lower_bound is None else round(lower_bound, FLOAT_PRECISION)
        self.check_lower_bound()
        self._upper_bound = self.value if upper_bound is None else round(upper_bound, FLOAT_PRECISION)
        self.check_upper_bound()

    def get_sp_dict(self):
        return {"time": self.time, "value": self.value,
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound}

    def __str__(self):
        return json.dumps(self.get_sp_dict())

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, val):
        assert val <= self.value, f"lower bound = {val} is larger than value = {self.value}"
        self._lower_bound = val

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, val):
        assert val >= self.value, f"upper bound = {val} is smaller than value = {self.value}"
        self._upper_bound = val

    def check_lower_bound(self):
        assert self.lower_bound <= self.value, f"lower bound = {self.lower_bound} is larger than value = {self.value}"

    def check_upper_bound(self):
        assert self.upper_bound >= self.value, f"upper bound = {self.upper_bound} is smaller than value = {self.value}"


class SafetyLimit:
    def __init__(self, lower_bound: float = None,
                 upper_bound: float = None):
        if lower_bound is None and upper_bound is None:
            raise ValueError(f"Not a valid SafetyLimit")
        if lower_bound is not None and upper_bound is not None:
            assert lower_bound <= upper_bound, f"lower bound = {lower_bound} should be less than or equal to " \
                                               f"upper bound = {upper_bound}"
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class Recipe:
    def __init__(self, sp_list: List[Dict[str, float]],
                 name: str):
        assert len(sp_list) > 0, "Can't initiate a recipe instance with an empty setpoint list"
        assert isinstance(name, str), "Can't initiate a recipe instance without a name"
        self.name = name
        self._sp_list = []
        self.index_lookup = {}
        self.add_setpoints(sp_list)

    @property
    def sp_list(self):
        return self._sp_list

    @sp_list.setter
    def sp_list(self, val: List[Setpoint]):
        self._sp_list = sorted(val, key=lambda x: x.time)
        self.index_lookup = self.create_index_lookup(self._sp_list)

    def _create_parameter_name(self, setpoint):
        return f"{self.name}@{setpoint.time}"

    @staticmethod
    def tokenize_parameter_name(parameter_name):
        tokens = parameter_name.split('@')
        recipe_name = tokens[0]
        str_time = tokens[1]
        return recipe_name, str_time

    @classmethod
    def create_from(cls, recipe_dict: Dict):
        assert len(recipe_dict) > 0, "recipe dict is empty"
        sp_list = []
        recipe_name = None
        for k, v in recipe_dict.items():
            parameter_name, time = cls.tokenize_parameter_name(k)
            if recipe_name is None:
                recipe_name = parameter_name
            assert recipe_name == parameter_name, f"recipe name should be consistent, " \
                                                  f"but found '{recipe_name}' and '{parameter_name}'"
            sp_list.append({"time": int(time), "value": v})
        return cls(sp_list, recipe_name)

    def dump(self) -> Dict:
        recipe_dict = {}
        for sp in self.sp_list:
            recipe_dict[self._create_parameter_name(sp)] = sp.value
        return recipe_dict

    def are_setpoints_in_safety_limit(self, safety_limit: SafetyLimit) -> bool:
        return all([self.is_setpoint_in_safety_limit(sp, safety_limit)
                    for sp in self.sp_list])

    @staticmethod
    def is_setpoint_in_safety_limit(sp: Setpoint, safety_limit: SafetyLimit) -> bool:
        lower_bound = safety_limit.lower_bound
        upper_bound = safety_limit.upper_bound

        if lower_bound is not None and upper_bound is not None:
            return safety_limit.lower_bound <= sp.value <= safety_limit.upper_bound

        elif lower_bound is not None and upper_bound is None:
            return safety_limit.lower_bound <= sp.value

        elif lower_bound is None and upper_bound is not None:
            return sp.value <= safety_limit.upper_bound

    @classmethod
    def create_index_lookup(cls, sp_list: List[Setpoint]) -> Dict:
        return dict([(sp.time, i) for i, sp in enumerate(sp_list)])

    def update_setpoint(self, sp: Setpoint):
        idx = self.index_lookup[sp.time]
        self.sp_list = self.sp_list[0:idx] + [sp] + self.sp_list[idx + 1:]

    def delete_setpoint(self, time: float):
        assert time in self.index_lookup.keys(), f"unable to delete non-existing setpoint, time = {time}"

        idx = self.index_lookup[time]
        self.sp_list = self.sp_list[0:idx] + self.sp_list[idx + 1:]

    def delete_setpoints(self, time_list: List[float]):
        for time in time_list:
            self.delete_setpoint(time)

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

    @staticmethod
    def shrink_by(value: float, safety_limit: SafetyLimit) -> float:
        lower_bound = safety_limit.lower_bound
        upper_bound = safety_limit.upper_bound

        if lower_bound is not None:
            value = max(value, lower_bound)

        if upper_bound is not None:
            value = min(value, upper_bound)

        return value

    def get_value_at(self, time: float,
                     value_type: ValueType = ValueType.NORMAL,
                     float_precision: int = FLOAT_PRECISION,
                     safety_limit: SafetyLimit = None,
                     filling_method: FillingMethod = FillingMethod.INTERPOLATION) -> float:

        left_sp, right_sp = self.find_setpoints_interval(time)

        if value_type == ValueType.NORMAL:
            left_val, right_val = left_sp.value, right_sp.value

        elif value_type == ValueType.LOWER_BOUND:
            left_val, right_val = left_sp.lower_bound, right_sp.lower_bound

        elif value_type == ValueType.UPPER_BOUND:
            left_val, right_val = left_sp.upper_bound, right_sp.upper_bound

        else:
            raise Exception(f"unknown value type {value_type}")

        val = -1

        if filling_method == FillingMethod.FROWARD:
            val = left_val

        elif filling_method == FillingMethod.BACKWARD:
            val = right_val

        elif filling_method == FillingMethod.INTERPOLATION:
            val_delta = right_val - left_val
            time_delta = right_sp.time - left_sp.time
            val = left_val if time_delta == 0 else \
                (val_delta / time_delta) * (time - left_sp.time) + left_val

            if safety_limit is not None:
                val = self.shrink_by(value=val, safety_limit=safety_limit)

        return round(val, float_precision)

    def get_points(self, bfill_from: float = None, ffill_to: float = None,
                   value_type: ValueType = ValueType.NORMAL,
                   safety_limit: SafetyLimit = None) -> List[List]:

        time_list = list(self.index_lookup.keys())

        assert len(time_list) > 0, "no setpoints available"

        if bfill_from is not None:
            assert bfill_from <= time_list[0], "Can't back-fill points to the given time, " \
                                               "it must be less than or equals to the time of the first setpoint"
            if time_list[0] != bfill_from:
                time_list = [bfill_from] + time_list

        if ffill_to is not None:
            assert ffill_to >= time_list[-1], "Can't forward-fill points to the given time, " \
                                              "it must be greater than or equals the time of the last setpoint"
            if time_list[-1] != ffill_to:
                time_list = time_list + [ffill_to]

        points_list = []

        for time in time_list:
            points_list.append([time, self.get_value_at(time=time, value_type=value_type,
                                                        safety_limit=safety_limit)])

        return points_list

    def get_lower_bound_points(self, safety_limit: SafetyLimit = None) -> List[List]:
        return self.get_points(value_type=ValueType.LOWER_BOUND,
                               safety_limit=safety_limit)

    def get_upper_bound_points(self, safety_limit: SafetyLimit = None) -> List[List]:
        return self.get_points(value_type=ValueType.UPPER_BOUND,
                               safety_limit=safety_limit)

    def get_search_space(self, safety_limit: SafetyLimit):
        search_space = []

        for setpoint in self.sp_list:
            lower_bound = max(safety_limit.lower_bound, setpoint.lower_bound)
            upper_bound = min(safety_limit.upper_bound, setpoint.upper_bound)
            search_space.append({
                "name": self._create_parameter_name(setpoint),
                "range": [lower_bound, upper_bound],
                "type": "real",
                "decimal": FLOAT_PRECISION
            })
        return search_space

    @classmethod
    def create_recipe_lookup_from_dump(cls, recipe_dump: Dict):
        recipe_dump_lookup = {}
        for parameter_name, value in recipe_dump.items():
            recipe_name, _ = cls.tokenize_parameter_name(parameter_name)
            if recipe_name in recipe_dump_lookup:
                recipe_dump_lookup[recipe_name][parameter_name] = value
            else:
                recipe_dump_lookup[recipe_name] = {parameter_name: value}

        for recipe_name, recipe_dump in recipe_dump_lookup.items():
            recipe_dump_lookup[recipe_name] = Recipe.create_from(recipe_dump)
        return recipe_dump_lookup


class RecipeCombo:
    def __init__(self, recipe_dict: Dict[str, Recipe]):
        self.recipe_dict = recipe_dict

    def get_values_dict_at(self, time: float,
                           value_type: ValueType = ValueType.NORMAL,
                           float_precision: int = FLOAT_PRECISION,
                           safety_limit: SafetyLimit = None) -> Dict:
        values_dict = {}
        for name, recipe in self.recipe_dict.items():
            values_dict[name] = recipe.get_value_at(time=time / MINUTES_IN_AN_HOUR,
                                                    value_type=value_type,
                                                    float_precision=float_precision,
                                                    safety_limit=safety_limit,
                                                    filling_method=FillingMethod.BACKWARD)

        return values_dict
