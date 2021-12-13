ENV["PYTHON"] = ""  # modify to the python path in conda

import Pkg
Pkg.build("PyCall")

