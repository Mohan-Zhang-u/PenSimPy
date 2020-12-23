FLOAT_PRECISION = 3
DEFAULT_MIN_TIME_IN_HOUR = 0
DEFAULT_MAX_TIME_IN_HOUR = 230
MINUTES_IN_AN_HOUR = 60

FS = "Fs"
FOIL = "Foil"
FG = "Fg"
PRES = "pressure"
DISCHARGE = "discharge"
WATER = "Fw"
PAA = "Fpaa"

DEFAULT_PENICILLIN_RECIPE_ORDER = [FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA]

FS_DEFAULT_PROFILE = [
    {"time": 3, "value": 8},
    {"time": 12, "value": 15},
    {"time": 16, "value": 30},
    {"time": 20, "value": 75},
    {"time": 24, "value": 150},
    {"time": 28, "value": 30},
    {"time": 32, "value": 37},
    {"time": 36, "value": 43},
    {"time": 40, "value": 47},
    {"time": 44, "value": 51},
    {"time": 48, "value": 57},
    {"time": 52, "value": 61},
    {"time": 56, "value": 65},
    {"time": 60, "value": 72},
    {"time": 64, "value": 76},
    {"time": 68, "value": 80},
    {"time": 72, "value": 84},
    {"time": 76, "value": 90},
    {"time": 80, "value": 116},
    {"time": 160, "value": 90},
    {"time": 230, "value": 80}
]

FOIL_DEFAULT_PROFILE = [
    {"time": 4, "value": 22},
    {"time": 16, "value": 30},
    {"time": 56, "value": 35},
    {"time": 60, "value": 34},
    {"time": 64, "value": 33},
    {"time": 68, "value": 32},
    {"time": 72, "value": 31},
    {"time": 76, "value": 30},
    {"time": 80, "value": 29},
    {"time": 230, "value": 23},
]

FG_DEFAULT_PROFILE = [
    {"time": 8, "value": 30},
    {"time": 20, "value": 42},
    {"time": 40, "value": 55},
    {"time": 90, "value": 60},
    {"time": 200, "value": 75},
    {"time": 230, "value": 65}
]

PRESS_DEFAULT_PROFILE = [
    {"time": 12.4, "value": 0.6},
    {"time": 25, "value": 0.7},
    {"time": 30, "value": 0.8},
    {"time": 40, "value": 0.9},
    {"time": 100, "value": 1.1},
    {"time": 150, "value": 1},
    {"time": 200, "value": 0.9},
    {"time": 230, "value": 0.9},
]

DISCHARGE_DEFAULT_PROFILE = [
    {"time": 100, "value": 0},
    {"time": 102, "value": 4000},
    {"time": 130, "value": 0},
    {"time": 132, "value": 4000},
    {"time": 150, "value": 0},
    {"time": 152, "value": 4000},
    {"time": 170, "value": 0},
    {"time": 172, "value": 4000},
    {"time": 190, "value": 0},
    {"time": 192, "value": 4000},
    {"time": 210, "value": 0},
    {"time": 212, "value": 4000},
    {"time": 230, "value": 0}
]

WATER_DEFAULT_PROFILE = [
    {"time": 50, "value": 0},
    {"time": 75, "value": 500},
    {"time": 150, "value": 100},
    {"time": 160, "value": 0},
    {"time": 170, "value": 400},
    {"time": 200, "value": 150},
    {"time": 230, "value": 250}
]

PAA_DEFAULT_PROFILE = [
    {"time": 5, "value": 5},
    {"time": 40, "value": 0},
    {"time": 200, "value": 10},
    {"time": 230, "value": 4}
]
