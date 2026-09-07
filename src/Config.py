####  IMPORTANT - DO NOT MODIFY CONFIG HERE
# This configuration is the default configuration and should not be altered
# Instead, modify the Config in your main file from which you start your experiments
#
# This can be done like:
# import Config

# Config.FEM_TYPE = "DG"
# Config.FEM_DEGREE = 2
####
# Note, that this has to happen BEFORE OTHER FILES ARE IMPORTED
# It will change the config in this namespace so consecutive imports
# can work on the changed configuration
###############


# Compiler options
from pathlib import Path

JIT_OPTIONS = {
    "cffi_extra_compile_args": ["-Ofast", "-march=native"],
    "cache_dir": f"{str(Path.cwd())}/.cache",
    "cffi_libraries": ["m"],
}

WAVE_PROPAGATION_SPEED = 1.0
FEM_TYPE = "Lagrange"
FEM_DEGREE = 1

DOF_TOLERANCE = 1e-9

# Linear Solvers
DIRECT = "direct"
ITERATIVE = "iterative"
ITERATIVE_METHOD = "cg"
PRECONDITIONER = "icc"
MAX_ITERATIONS = 2000
ABS_TOLERANCE = 1e-12
REL_TOLERANCE = 1e-12
DEFAULT_SOLVING_TYP = DIRECT

# names and TeX
TAU = "tau"
TAU_TEX = "$\\tau$"
RATE_1_TIME = "$\\tau$"
RATE_2_TIME = "$\\tau^2$"
RATE_3_TIME = "$\\tau^3$"
RATE_2_SPACE = "$h^2$"

# Time Integrators Strings
DS_STR = "DomainSplitting"
DSLP_STR = "DomainSplittingLocalizedPrediction"
DSTLP_STR = "DomainSplittingTestLocalizedPrediction"
DSE_STR = "DomainSplittingExtrapolation"
DSMPR_STR = "DomainSplittingMultiRatePrediction"
CN_STR = "CrankNicolson"
LF_STR = "leapfrog"
ATI_STR = "AbstractTimeIntegrator"
ATI_2ND_STR = "AbstractTimeIntegratorSecondOrder"
ATI_DG_STR = "AbstractTimeIntegratorDG"
ATI_WEAK_BC_STR = "AbstractTimeIntegratorWeakBC"
CN_DG_STR = "CrankNicolsonDG"
LF_DG_STR = "leapfrogDG"
DS_DG_STR = "DomainSplittingDG"


# errors
H1ERROR_Q_STR = "H1 error q"
L2ERROR_Q_STR = "L2 error q"
L2ERROR_P_STR = "L2 error p"
HMERROR_P_STR = "H-1 error p"
REL_ERROR_PREFIX = "relative "
REL_H1L2_ERROR_STR = "relative H1L2 error"
REL_L2HM1_ERROR_STR = "relative L2Hm1 error"
TIME_STR = "wall time"
